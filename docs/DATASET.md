# Dataset

## Source and licence

| Item | Value |
|---|---|
| Name | "site dataset" — person detection, version 1 |
| Provider | Roboflow Universe user "Domin" |
| URL | <https://universe.roboflow.com/domin-td1ad/site-dataset-vrazk> |
| Licence | CC BY 4.0 |
| Export | COCO format, exported via roboflow.com on July 1, 2026 (see `annotations/README.roboflow.txt`) |

**Attribution.** The annotation files in `annotations/` are redistributed unchanged from the Roboflow export above
under CC BY 4.0. No corrections were made, including for the annotation loss described below.

**Images are not included.** Download the same export (version 1, COCO format) from the provider and copy the image
files into `annotations/train/`, `annotations/valid/` and `annotations/test/`, next to each `_annotations.coco.json`.
Image use is subject to the provider's terms.

## Splits

| Split | Images | Distinct source frames | Person boxes |
|---|---:|---:|---:|
| Train | 1,425 | 475 | 5,536 |
| Valid | 161 | 161 | 831 |
| Test | 94 | 94 | 457 |
| **Total** | **1,680** | **730** | **6,824** |

- The export contains one foreground category (`person`, id 1) and a dummy super-category (`site-dataset`), which is
  ignored.
- Images are mostly 1920×1080 (1,104 of 1,680). Most of the rest are 2048×1152; a few are portrait or smaller frames.

## Provider pre-processing and augmentation

- **Pre-processing:** auto-contrast by histogram equalisation, applied to every image.
- **Offline augmentation:** applied to the training split only. Each source image has three augmented copies, produced
  with these operations:
  - horizontal and vertical flips (each with probability 0.5);
  - a random 90° rotation (none, clockwise, counter-clockwise or upside-down);
  - brightness adjustment of ±15% and exposure adjustment of ±10%;
  - salt-and-pepper noise on 0.1% of pixels.

## Annotation-loss audit

The provider's 90° rotations are letterboxed into the original landscape canvas, and many person annotations are
missing in those copies:

- 207 of the 475 source frames have copies with different box counts.
- 836 boxes are missing relative to the best-annotated copy of the same source frame. 825 of these are missing from
  letterboxed copies.
- 65 of the 164 training images without annotations have an annotated sibling copy.

The provider-side cause is unknown. All results use the released annotations without correction.

## SHA-256 of the annotation files

```text
6b5072086d473246a891c7a7d3806f7d981338ab3939e14946a1d8060b5d9f42  annotations/train/_annotations.coco.json
599e95879c4be5d4a5c608ae6414bbc866add9db757edb177053802e9ce5616c  annotations/valid/_annotations.coco.json
83d492dac4c5ad77edf75fe1c6b8b3bb3a40cea61c8837bc5655187488debe82  annotations/test/_annotations.coco.json
```
