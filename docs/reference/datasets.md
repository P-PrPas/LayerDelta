# Datasets and declarative preprocessing (v0.5-E2)

Design: [v0.5 design §2](../architecture/v0.5-evaluators-benchmark.md). Code: `datasets.py` (import, manifest, splits, spec check; stdlib only), `preprocess.py` (decode and apply the spec; torch and Pillow, evaluation worker only).

A dataset is a **human-only import**, like a checkpoint. Agents list and read registered datasets and never name a path.

```
POST /dataset-imports {path, format, name, preprocessing?}   human only, 202 {import_id}     two imports at a time
GET  /dataset-imports/{import_id}                             human only: status, dataset | error
GET  /datasets, /datasets/{dataset_id}                        model:read; tools list_datasets / get_dataset (also in the structure profile)
```

Import reads the directory as **data**: it hashes every file, parses the PNG or BMP **header** (dimensions, bit depth), and copies the file to `datasets/files/<sha256>`. It never decodes pixels, so a hostile image cannot reach the server process. Paths in the index must be relative names of regular files inside the directory (no `..`, no absolute path, no symlink). Limits (`Limits`): 4 to 200,000 samples, 64 MiB per file, 8 GiB in total, 2^26 pixels and 16,384 on a side. Errors: `unknown_format`, `invalid_name`, `invalid_path`, `invalid_index`, `invalid_preprocessing`, `unsupported_image`, `dataset_too_small`, `dataset_too_large`.

## Formats (a closed set)

| Format | Task | Files |
|---|---|---|
| `image-folder-labels` | classification | `labels.csv` (`image,label[,condition]`) and the images; classes are the sorted labels (2 to 4096) |
| `image-mask` | segmentation | `index.csv` (`image,mask[,condition]`), `classes.json` `{"classes": [...], "ignore_index": 255}`; masks are 8-bit grayscale or palette PNG of the image's size |
| `image-boxes` | detection | `annotations.json`, a COCO subset: `images` (`id, file_name, width, height[, condition]`), `annotations` (`image_id, bbox [x, y, w, h], category_id`), `categories`; boxes must lie inside their image |
| `image-text` | ocr | `index.csv` (`image,text[,condition]`); text is printable ASCII up to 256 characters; the alphabet is the characters used |

Images are **PNG** (8-bit, not interlaced) or **BMP** (uncompressed 8, 24 or 32 bit). There is no JPEG in v0.5: decoders differ by libjpeg version. A sample id is the image's relative path without its extension (`[A-Za-z0-9._-]`, at most 64 characters, unique). A new format is a code change in `src/`, with no model-specific code.

## Splits and what an agent sees

`exploration` and `holdout` by the v0.1 rule `stratified-hash-half-v1`: per stratum (the label for classification, the condition otherwise), samples are ranked by SHA-256 of their id and the first half is exploration (the odd one goes to holdout). The dataset id is `ds_` plus a hash of the manifest and the preprocessing, so the same content with a different preprocessing is a different dataset. Importing the same thing again returns the same dataset.

A person sees the whole record. A token sees `counts` and `split_sha256` for **exploration only**: the holdout count, hash and rows are never in an agent's view. `mnist-v1` is the first dataset, the v0.1 manifest unchanged (its samples come from the idx archives).

## Preprocessing

A closed spec, applied in this fixed order: `resize {size [h, w], mode nearest|bilinear, antialias}` or `resize_height {height, keep_aspect: true}` (a CRNN-style line: the width follows the aspect), `pad_to {size, value, anchor top_left|center}`, `grayscale` (Rec. 601 luma), `channel_order rgb|bgr`, `scale`, then `mean` and `std` (together, 1 or 3 values). `{}` is the identity: float32 pixels 0 to 255.

One implementation: `torch.nn.functional.interpolate`; PIL resampling is never used, because pixels differ across libraries and versions. `preprocess.implementation_version()` (preprocess version, torch, Pillow) goes into an evaluation's fingerprint, so an upgrade cannot change results silently.

**Labels follow the geometric steps.** A mask is resized with nearest-neighbour and padded with the ignore index. Boxes are scaled and offset, clipped to the output, and dropped when they become empty (`box_index` says which rows were kept). Text is unchanged. `geometry` records `{source, scale, offset, output}`, and `to_source(boxes, geometry)` maps boxes back to source-image coordinates.

**Decoding** runs on untrusted bytes, so it happens only in the evaluation worker, with Pillow limited to PNG and BMP, the pixel cap above, and the header check repeated first. Pillow is the optional extra `layerdelta[eval]`.
