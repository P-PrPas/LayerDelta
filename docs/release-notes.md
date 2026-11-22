# LayerDelta 0.1.0rc1

> **Internal build, not published.** No tag or public release exists, and publication is parked until the owner explicitly requests it ([Roadmap](Roadmap.md)).

This candidate supports the two-block MNIST CNN on Windows x64. Users can inspect a paired source image, scale or zero one post-ReLU activation or convolution kernel, and compare before/after predictions by sample ID. The desktop workbench and headless Agent share a loopback service. Human confirmations govern holdout evaluation and export of weight-derived Safetensors children.

## Evidence

- [M0 reference fixture](m0-reference.md) records artifact identities, class counts, preprocessing and the demonstration slice.
- [M1 runner measurements](m1-measurements.md) record CPU/CUDA agreement, calibration, timing, memory and recovery from an interrupted worker.
- [M2 workflow](m2-measurements.md) records real replay, changed samples, exported child identity and checks for both runtimes.
- The [Windows quickstart](windows-quickstart.md) covers both installation profiles. The [adapter guide](adapter-guide.md) describes the mapping and tests required to add another topology.

## Distribution

LayerDelta's code is licensed under [Apache-2.0](../LICENSE). The package includes notices for its bundled software. Users download the four MNIST archives from the CVDF mirror and generate the demonstration checkpoint locally. The release package does not redistribute those images or the trained weights. [Artifact provenance](m0-reference.md)
