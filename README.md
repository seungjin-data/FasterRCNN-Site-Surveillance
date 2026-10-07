# Fine-Tuning Faster R-CNN for Person Detection in Site Surveillance Imagery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23214797.svg)](https://doi.org/10.5281/zenodo.23214797)

Reproducibility package for the Machine Vision and Applications manuscript.

**Author:** S. J. Kim, Institute for Industrial Policy Studies (IPS), aSSIST University, Seoul, Republic of Korea
([ORCID 0009-0007-6876-0777](https://orcid.org/0009-0007-6876-0777))

**Manuscript:** *Fine-Tuning Faster R-CNN for Person Detection in Site Surveillance Imagery: A Size-Stratified
Evaluation Against an Off-the-Shelf COCO-Pretrained Baseline* (in preparation for submission to *Machine Vision and
Applications*).

## Overview

This repository supports a controlled empirical machine-vision study of domain-specific fine-tuning for
site-surveillance person detection. It does not propose a new detector architecture. A ResNet-50-FPN Faster R-CNN
(torchvision) is fine-tuned on a public single-class site dataset (1,680 images, 6,824 annotated persons) and compared
with the identical off-the-shelf COCO-pretrained model. The analysis covers:

- size-stratified COCO metrics;
- four torchvision detector families plus Ultralytics YOLOv8m and RT-DETR-L reference runs;
- seed variability and ablations;
- paired image-level bootstrap uncertainty;
- an exploratory proxy scene-held-out analysis;
- an audit of the dataset's offline augmentation.

## Main findings

The test split has 94 images and was never used for model selection. Results are for the validation-selected
checkpoint (epoch 65 of a 100-epoch schedule).

| Metric | Off-the-shelf COCO-pretrained | Fine-tuned |
|---|---|---|
| AP (COCO, IoU 0.50:0.95) | 0.302 | 0.455 |
| AP50 | 0.628 | 0.817 |
| APS (small persons) | 0.162 | 0.359 |

- **Paired bootstrap gain:** ΔAP = +0.153, 95% CI 0.120–0.184 (B = 1,000, image-level resampling).
- **Seed variability:** under the 15-epoch recipe, three seeds give AP 0.438 ± 0.007.

## Repository contents

```text
FasterRCNN-Site-Surveillance/
├── README.md               this file
├── LICENSE                 licence statement (see "Licence")
├── CITATION.cff            citation metadata
├── requirements.txt        Python dependencies
├── core/                   model, dataset, training and evaluation code (PyTorch/torchvision)
├── paper_scripts/          evaluation, bootstrap, scene clustering/split, Ultralytics runs,
│                           figure generation, run queues (run_*.sh); all paths via _paths.py
├── results/                per-model results (*.json), stored test-set detections (*_test_dets.json),
│                           bootstrap outputs, scene clusters and evaluation, main training log and history
├── annotations/            COCO-format annotations of the train/valid/test splits + provider READMEs
├── inference_example/      standalone inference script for the main model
├── figures/                Fig. 1–7 of the manuscript (PDF)
└── docs/
    ├── REPRODUCIBILITY.md  step-by-step reproduction and provenance notes
    ├── DATASET.md          dataset source, licence and split statistics
    ├── MODEL_WEIGHTS.md    trained weights: filenames, SHA-256, upstream models, redistribution status
    └── SHA256SUMS.txt      SHA-256 manifest of every file in this repository
```

## Reproduction

The detailed steps are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

1. **Environment setup.** `pip install -r requirements.txt`. The study used PyTorch 2.5.1 and torchvision 0.20.1;
   install a CUDA build from <https://pytorch.org> for GPU training.
2. **Dataset annotations.** The COCO-format annotation files are in `annotations/`. Download the images of the same
   export from the provider (see [`docs/DATASET.md`](docs/DATASET.md)) and place them next to the annotation files.
3. **Evaluation from stored detections.** The test-set metrics are recomputed from `results/*_test_dets.json` and the
   annotations. No GPU or images are needed.
4. **Bootstrap.** `python paper_scripts/bootstrap_ci.py 1000`.
5. **Figure generation.** `python paper_scripts/make_figures_v2.py pipeline dataset training pr size tradeoff` writes
   Fig. 1–6. Fig. 7 (`make_qualitative.py`) needs the test images and the main model weights.
6. **Optional model inference.** See [`inference_example/`](inference_example/) and
   [`docs/MODEL_WEIGHTS.md`](docs/MODEL_WEIGHTS.md).

Training uses `torch.manual_seed` with cuDNN benchmark mode. Retrained models are therefore not bit-identical to the
reported runs.

## Dataset

- The study uses the public Roboflow Universe "site dataset" (person detection, version 1), licensed CC BY 4.0.
- Raw images are not redistributed in this repository. Obtain them from the original provider.
- The COCO-format annotation files of the three splits are included, as the CC BY 4.0 licence permits, with
  attribution in [`docs/DATASET.md`](docs/DATASET.md).

## Model weights

Large binary weights are not stored in this repository. [`docs/MODEL_WEIGHTS.md`](docs/MODEL_WEIGHTS.md) lists each
trained weight file with its SHA-256, upstream model and redistribution status. The Zenodo archive v1.0.0
(https://doi.org/10.5281/zenodo.23214797) contains no weight files; weights whose redistribution is permitted may be
added to a later archive version.

## Limitations

- The historical scene-held-out checkpoint was not preserved, so that model cannot be independently re-inferred. The
  split definition (`results/scene_clusters.json`) and summary results (`results/scene_eval.json`) are preserved.
- The scene-held-out result is an exploratory sensitivity analysis, not an unbiased estimate of unseen-site
  generalisation loss:
  - the held-out model was trained on about 21% fewer images (1,119 versus 1,425);
  - the clusters are appearance-based proxies rather than camera or site identifiers;
  - the criterion used to select the held-out clusters was not preserved.
- The provider's letterboxed 90° rotated training copies show substantial person-annotation loss:
  - 836 boxes are missing relative to the best-annotated sibling copy, and 825 of these are associated with
    letterboxed copies;
  - 65 of the 164 empty training images have an annotated sibling.

  The original annotations were used without correction. The provider-side cause is unknown.

## Licence

The source code written for this study (`core/`, `paper_scripts/`, `inference_example/`) is released under the
[MIT License](LICENSE). Third-party datasets, pretrained model weights, and external software remain subject to their
respective licence and usage terms (CC BY 4.0 for the dataset annotations; torchvision and Ultralytics terms for
pretrained weights and software). See [`LICENSE`](LICENSE) for details.

## Citation

Versioned archive of this repository (v1.0.0) on Zenodo: <https://doi.org/10.5281/zenodo.23214797>

> Kim, S. J. (2026). *Reproducibility Package for Fine-Tuning Faster R-CNN for Person Detection in Site Surveillance
> Imagery* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23214797

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).
