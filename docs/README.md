# Documentation

Start with the [usage guide](usage.md) for installation, configuration, output
formats and label review. The [examples](../examples/README.md) provide runnable
configurations; [validation evidence](../benchmarks/README.md) describes what has
been measured and where the decoder abstains.

| Task | Guide |
|---|---|
| Configure segments, gaps and group boundaries | [Blocked grids](blocked-grids.md) |
| Configure extraction and review identifiers | [Usage](usage.md) |
| Propose regions/counts and review automation decisions | [Automation](automation.md) |
| Review colourbar ticks | [Legend review](legend.md) |
| Choose a reviewed numeric model and inspect gradients | [Calibration](nonlinear-calibration.md) |
| Understand colour ambiguity and clipping | [Palettes](palettes.md) |
| Read arbitrary-angle sample names | [Arbitrary angles](arbitrary-angle-labels.md) |
| Preserve exact quarter-turn paths | [Rotated labels](rotated-labels.md) |
| Review orientation proposals | [Orientation](orientation-proposals.md) |
| Understand geometry corrections | [Corrections](corrections.md) |

[Specifications](specs/) describe implementation requirements. [Milestone logs](progress/)
retain historical decisions and measurements; they are not the primary user guide.
[M16 status](progress/M16.md) consolidates the current implementation and its
remaining source-data limitations. Earlier M16 trial narratives are explicitly
[archived](progress/archive/M16-initial-and-followups.md).

[M17 status](progress/M17.md) records blocked geometry and its current automation limits.

[M18 status](progress/M18.md) records angle transforms, attachment geometry and identifier review.

[M19 status](progress/M19.md) records capability decisions and remaining gates.
See [confident automation](confident-automation.md) for separate numeric/identifier states.
