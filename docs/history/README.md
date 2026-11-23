# History

The documents in this folder are **historical records**. They describe what was planned, measured or decided at a particular point, and they are not maintained as current guidance. For the current state, use the [roadmap](../roadmap/Roadmap.md) and the [architecture](../architecture/Architecture.md) documents.

Some of the evidence here is still referenced from current docs, for example the measurement records behind the v0.1 acceptance gates.

## Product requirements and planning (v0.1)

| Document | Purpose |
|---|---|
| [LayerDelta Open Source Proposal](LayerDelta_Open_Source_Proposal.md) | Draft proposal for the open-source project: problem, users, MVP scope and risks |
| [PRD v0.1](LayerDelta_PRD_v0.1.md) | Product requirements for the v0.1 reference workbench, with the functional requirements and release gates G1–G10 |
| [Implementation plan v0.1](Implementation_Plan_v0.1.md) | Milestones M0–M4 for v0.1, with the acceptance evidence table |
| [Agent extension spec](LayerDelta_Agent_Extension.md) | Parked research spec for weight inspection by agents, experiment trace and a benchmark (now v0.2 and v0.5 work) |
| [Human pilot protocol](human-pilot.md) | Parked protocol for a timed human pilot, to run during release readiness |

## Measurement and validation records (v0.1)

| Document | Purpose |
|---|---|
| [M0 reference measurements](m0-reference.md) | Reference fixture identities, class counts, preprocessing and the demonstration slice |
| [M0 desktop packaging spike](m0-desktop.md) | Measurements for the Windows desktop shell choice and startup time |
| [M1 runner measurements](m1-measurements.md) | CPU/CUDA agreement, calibration, timing, memory and recovery of the runner |
| [M2 HTTP workflow](m2-measurements.md) | Replay, changed samples, exported child identity and checks for both runtimes |
| [M3 beta validation](m3-validation.md) | Holdout, failure-path and scripted agent pilot results on Windows x64 |
| [M4 release candidate package](m4-package.md) | The tested Windows release ZIP and wheel: sizes, digests and packaging details |
| [M4 release candidate validation](m4-validation.md) | Installed-package validation on a Windows host with CPU and CUDA profiles |

## Release notes

| Document | Purpose |
|---|---|
| [Release notes 0.1.0rc1](release-notes.md) | Summary of the unpublished 0.1.0rc1 internal build: scope, evidence and distribution notes |
