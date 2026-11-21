# M0 reference measurements

## Fixture

The built-in CNN accepts a float32 grayscale image of shape `[1,28,28]`. Both blocks run a biased 3×3 convolution with padding 1, ReLU, then 2×2 max pooling. The blocks expose 8 and 24 output channels. The classifier flattens 24×7×7 into a linear layer with ten output logits. The generator records every target ID, tensor shape and editable channel axis in `fixture.json`.

| Target | Hook or parameter | First and last channel |
|---|---|---|
| features.block1.relu.output | post-ReLU output, axis 1 | 0, 7 |
| features.block1.conv.kernel | features.block1.conv.weight, axis 0 | 0, 7 |
| features.block2.relu.output | post-ReLU output, axis 1 | 0, 23 |
| features.block2.conv.kernel | features.block2.conv.weight, axis 0 | 0, 23 |

The fixture generator trains for one epoch on 60,000 normal training samples with seed 20261122 and CPU float32 arithmetic. The official test partition is stratified by digit and divided by source image before creating image variants. Exploration has 4,999 source images and final holdout has 5,001. Each source contributes a normal image and a blurred image using a 5×5 mean filter, stride 1 and zero padding. The choice of 5×5 yielded 47 inspectable examples of the most frequent error and was retained. The training and demo-selection paths evaluate training and exploration splits respectively.

## Frozen inputs and results

| Artifact | SHA-256 |
|---|---|
| model.safetensors | cf9e1bb4c660949dd574521c7436884e8f5d4abd0ae6f8679c4a7011f361ce69 |
| dataset.json (file bytes) | da1617b19a5ef1645bdfcc591b874efaaaa6211869cf3690b708f1ea62ee42ea |
| Dataset manifest identity (canonical payload) | beee8dcff3438db6f633412931030ae181e907e3fa6b12fcbd351938a283da42 |
| exploration.json | 71a2fa3430cb3462b513a1ab42953416289c0fde2a1988b4a523bc2f2492e9d2 |

Two independent CLI builds in separate output directories reproduced all three file hashes and the complete fixture metadata byte for byte on this machine.

On exploration, the CPU baseline classified **4,811/4,999 normal** and **4,598/4,999 blurred** images correctly. The most common off-diagonal entry in the blurred confusion matrix was **9 → 4**, affecting 47 of the 504 blurred images with true label 9. All 504 form the primary slice. All 4,999 normal images form the guardrail. The generated `exploration.json` contains both 10×10 confusion matrices and the IDs of five example pairs.

| Probe | Recorded configuration | Actual device | Prediction for source 00530, normal / blurred |
|---|---|---|---|
| CPU | Windows 11 x64 build 26200, Python 3.12.5, PyTorch 2.6.0+cu124 | cpu | 9 / 4 |
| CPU-only installation | Same Windows host, independent environment with PyTorch 2.6.0+cpu | cpu | 9 / 4 |
| CUDA | Same host and checkpoint, RTX 3050 Laptop GPU, 4 GiB, driver 566.07, CUDA wheel 12.4 | cuda:0 | 9 / 4 |

Each probe loaded the frozen checkpoint in a separate process. All read the same two exploration rows and returned finite logits. These forwards establish device availability for M0. M1 will measure tolerances on a calibration set and verify full intervention behavior on each runtime.

## Provenance and distribution

MNIST is attributed to Yann LeCun, Corinna Cortes and Christopher J. C. Burges. The generator retrieves the four compressed IDX files from the CVDF mirror and checks the published digests before loading. This PR distributes training code, source references, digests and measurements. Running the generator creates the dataset cache and locally trained checkpoint in the user's workspace. The owner will decide code licensing and review distribution terms for dataset and checkpoint separately before release. [MNIST dataset catalog](https://www.tensorflow.org/datasets/catalog/mnist) · [Archive checksums](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/mnist.txt)
