"""Bundled local RapidOCR, isolated in a reusable worker with a hard timeout."""

import atexit
import hashlib
import importlib
import importlib.metadata
import math
import pickle
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import OCRConfig, OCRRun, OCRToken

_worker: subprocess.Popen[bytes] | None = None
_lock = threading.Lock()


def _serve() -> None:
    """One model load per process; binary IPC never imports the caller's script."""
    try:
        pixels, detection = pickle.load(sys.stdin.buffer)
        module = importlib.import_module("rapidocr_onnxruntime")
        engine = module.RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        if module.__file__ is None:
            raise ValueError("RapidOCR package has no model root")
        root = Path(module.__file__).parent
        models = tuple(
            sorted(
                (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                for p in (root / "models").glob("*.onnx")
            )
        )
        version = importlib.metadata.version("rapidocr-onnxruntime")
        runtime = importlib.metadata.version("onnxruntime")
        while True:
            result, _ = engine(pixels, use_det=detection, use_cls=False, text_score=0)
            pickle.dump(
                {"result": result, "version": version, "runtime": runtime, "models": models},
                sys.stdout.buffer,
            )
            sys.stdout.buffer.flush()
            pixels, detection = pickle.load(sys.stdin.buffer)
    except EOFError:
        pass
    except Exception as error:
        try:
            pickle.dump({"error": f"{type(error).__name__}: {error}"}, sys.stdout.buffer)
            sys.stdout.buffer.flush()
        except (BrokenPipeError, EOFError, OSError):
            pass


def _stop_worker() -> None:
    global _worker
    if _worker is not None:
        process = _worker
        _worker = None
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()


atexit.register(_stop_worker)


def _request(pixels: np.ndarray[Any, Any], detection: bool, timeout: float) -> dict[str, Any]:
    global _worker
    started = time.monotonic()
    if not _lock.acquire(timeout=timeout):
        raise ValueError("Local RapidOCR timeout waiting for worker")
    try:
        if _worker is None:
            process = subprocess.Popen(
                [sys.executable, "-m", "heatmap_decoder.rapid_ocr"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            _worker = process
        process = _worker
        replies: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)

        def exchange() -> None:
            try:
                assert process.stdin is not None and process.stdout is not None
                pickle.dump((pixels, detection), process.stdin)
                process.stdin.flush()
                response: dict[str, Any] = pickle.load(process.stdout)
                replies.put(response)
            except (EOFError, BrokenPipeError, OSError, ValueError, pickle.PickleError) as error:
                replies.put({"error": f"worker IPC failed: {error}"})

        thread = threading.Thread(target=exchange, daemon=True)
        thread.start()
        try:
            result = replies.get(timeout=max(0, timeout - (time.monotonic() - started)))
        except queue.Empty as error:
            _stop_worker()
            thread.join(timeout=1)
            raise ValueError("Local RapidOCR timeout; worker terminated") from error
        thread.join(timeout=1)
        if "error" in result:
            _stop_worker()
            raise ValueError(f"Local RapidOCR failed: {result['error']}")
        return result
    finally:
        _lock.release()


def recognize_rapid(
    image: Image.Image, settings: OCRConfig, *, mode: str, timeout_seconds: float
) -> OCRRun:
    """Single-field recognition or detected regions; retain exact engine strings."""
    from .ocr import OCRUnavailableError

    if settings.language != "eng":
        raise OCRUnavailableError(
            "RapidOCR uses the bundled PP-OCRv4 model with language='eng'; "
            "select engine='tesseract' for other installed language packs"
        )
    try:
        importlib.metadata.version("rapidocr-onnxruntime")
    except importlib.metadata.PackageNotFoundError as error:
        raise OCRUnavailableError(
            "Install heatmap-decoder to obtain its RapidOCR dependency"
        ) from error
    detection = mode == "localisation"
    payload = _request(np.asarray(image.convert("RGB")), detection, timeout_seconds)
    tokens = []
    for index, record in enumerate(payload["result"] or []):
        if detection:
            points, text, score = record[:3]
            coordinates = np.asarray(points, dtype=float)
            if coordinates.shape != (4, 2) or not np.isfinite(coordinates).all():
                raise ValueError("RapidOCR returned invalid detected geometry")
            box = (
                max(0.0, float(coordinates[:, 0].min())),
                max(0.0, float(coordinates[:, 1].min())),
                min(float(image.width), float(coordinates[:, 0].max())),
                min(float(image.height), float(coordinates[:, 1].max())),
            )
        else:
            text, score = record[:2]
            box = (0.0, 0.0, float(image.width), float(image.height))
        score = float(score)
        if not isinstance(text, str) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("RapidOCR returned invalid text/score")
        if not text.strip():
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("RapidOCR returned empty detected geometry")
        tokens.append(OCRToken(text, box, score * 100, (1, index + 1, 1)))
    return OCRRun(
        tuple(tokens),
        f"PP-OCRv4 / ONNX Runtime {payload['runtime']}",
        payload["version"],
        settings.language,
        ("bundled-PP-OCRv4",),
        "rapidocr-detect-recognize" if detection else "rapidocr-recognize-field",
        engine_name="RapidOCR",
        geometry_kind="detected_text" if detection else "source_field",
        score_kind="uncalibrated_mean_character_score_x100",
        model_sha256=payload["models"],
    )


if __name__ == "__main__":
    _serve()
