# Repository instructions

## Start from the active development branch

**Do implementation work on `work`, based on the latest `origin/work`.**
The default branch currently is `main` (not `master`) and may contain only the
initial scaffold. An apparently empty checkout does not mean the application
needs to be created from scratch: the existing application lives on `work`.

Before planning, editing, installing a replacement project, or drawing
conclusions about missing code:

1. Read `git status --short --branch` and preserve any existing local changes.
2. Discover the remote branches and default branch:
   ```sh
   git ls-remote --symref origin HEAD
   git ls-remote --heads origin
   ```
3. Fetch `work` explicitly. Some task checkouts fetch only the default branch,
   so plain `git fetch origin` may not create or update `origin/work`:
   ```sh
   git fetch origin refs/heads/work:refs/remotes/origin/work
   ```
4. Switch to the existing local `work` branch, or create it tracking
   `origin/work` if absent. On a clean checkout, bring it up to date with
   `git merge --ff-only origin/work`. If local changes or divergent commits
   prevent that, preserve them and reconcile with the remote before implementing.
   Do not reset, clean, force-push, or overwrite other work to make this easier.
5. Inspect the latest commit, `README.md`, `pyproject.toml`, `src/`, `tests/`,
   and relevant existing examples. Read any more specific `AGENTS.md` files.

If `origin/work` is unavailable, report the actual fetch/access failure rather
than treating the default scaffold as the complete repository.

## Integrate with the existing application

- Extend the existing `src/heatmap_decoder` package and CLI. Do not introduce a
  competing root-level package that shadows it, or replace the project's build
  configuration, README, dependencies, or tests with a minimal new application.
- Add publication fixtures to `examples/real_world` and its catalogue. Preserve
  original images and source attribution. Distinguish verified plotted values,
  image-derived estimates, annotation references, and unresolved sample identities.
- Implement and test requested behavior; adding a fixture or research note alone
  is not implementation. Clearly identify work that remains unimplemented or
  unverified. Keep reproducible extraction results separate from ground truth.

## Verify and publish

- Follow the project's locked environment and checks in
  `.github/workflows/ci.yml`. Run tests appropriate to the change, including the
  relevant opt-in OCR tests when changing OCR behavior. Run Ruff and strict mypy
  for code changes; verify packaging/CLI behavior when changing those interfaces.
- Before publishing, fetch/check `origin/work` again and integrate any new remote
  changes. Push development commits to `work`; do not overwrite remote history.
- Only change or publish the default branch when the user explicitly requests it.
  A request to maintain this instruction file on the default branch is such an
  exception; it does not authorize merging application changes there.
- Keep this file available on both the default branch and `work`, and finish with
  the primary checkout on `work`.
- Respect GitHub email privacy protection. If a commit is rejected for a private
  email, use the same author's public GitHub no-reply identity already established
  in repository history for the unpushed commit. Do not disable account protections.
- Report the actual tests run, unresolved limitations, branch, and pushed commit.
