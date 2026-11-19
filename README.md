# MedGemma CT Fibrosis

Fine-tuning a medical vision-language model for pulmonary fibrosis detection on a consumer laptop.

> **Research use only. Not for clinical use.**
> This is a proof-of-concept trained on public datasets. It has not been clinically
> validated and must not be used for patient care or diagnostic decisions.

## Overview

This project fine-tunes Google's
[MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it), a
4-billion-parameter medical vision-language model, to classify lower-lobe chest CT
slices as showing signs of pulmonary fibrosis or not. All training and evaluation
were carried out on a single consumer laptop.

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM |
| Training time | 34 minutes |
| Training data | 103 fibrosis patients, 169 normal |
| Patient-level accuracy, neutral prompt | 76.7% (base model) to 88.9% (fine-tuned) |

- Model weights: [`mku1988/medgemma-ct-fibrosis`](https://huggingface.co/mku1988/medgemma-ct-fibrosis) <!-- TODO: confirm repo name -->
- Demo: Gradio application in [`app/`](app/)

## Results

The model was evaluated on 81 held-out patients (32 fibrosis, 49 normal) who were
not used in training. Each patient was classified by majority vote over 7 lower-lobe
slices. All patients received the same neutral prompt, so the prediction depends on
the image alone. This matches how the model is used in practice.

| Metric | MedGemma 1.5 (base) | Fine-tuned |
|---|---|---|
| Accuracy | 76.7% | 88.9% |
| Sensitivity | n/a | 90.6% (29 of 32) |
| Specificity | n/a | 87.8% (43 of 49) |
| Responses in the required output format | 0% | 98 to 100% |

**Run-to-run variation.** Inference with 4-bit weights on a GPU is not bit-for-bit
deterministic, so a few borderline slices can change their vote between runs. In
three independent evaluations of the released model, patient-level accuracy was
88.9%, 88.9% and 90.1% (a difference of one patient). The table reports the lower
value.

**Robustness.** The model was also tested with deliberately misleading clinical
text: fibrosis patients described as routine screening cases, and normal patients
described with a disease-style history. Accuracy under these conditions was 85.2%,
compared with 77.9% for the base model.

## Training on modest hardware

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop, 8 GB VRAM (peak use approximately 7.4 GB) |
| Training patients | 272 (103 fibrosis, 169 normal) |
| Training time | 33 min 42 s (4 epochs) |
| Method | 4-bit QLoRA with [Unsloth](https://github.com/unslothai/unsloth); vision encoder frozen |
| Scope | Data preparation, all experiments, evaluation and the demo ran on the same laptop. No cloud compute was used. |

Three choices made this feasible:

1. **Unsloth with 4-bit quantisation.** The base model is loaded in 4-bit precision
   and only small LoRA adapters are trained. This brings a model that would
   normally require a data-centre GPU within an 8 GB memory budget.
2. **Frozen vision encoder.** MedGemma's image encoder is already trained on medical
   images. Keeping it fixed reduces memory use and preserves that prior knowledge,
   which a few hundred training examples could otherwise degrade.
3. **Dynamic slice sampling.** Each training example contains one slice. A different
   lower-lobe slice is drawn for each patient in every epoch, which increases the
   variety of training images at no additional memory cost.

**Outlook.** These results were obtained with roughly one hundred fibrosis patients.
In this setting the main constraint was the amount and diversity of data rather
than compute. We expect that a larger dataset, and in particular one in which
fibrosis and normal scans come from the same hospitals and scanners, would improve
both accuracy and reliability while remaining trainable on similar hardware. This
expectation has not yet been tested.

## Development notes

The final model was reached through a sequence of experiments. The main findings
are summarised below.

**Perfect test scores required investigation.** An early fine-tuned model scored
100% on the test set. Analysis showed that it was classifying patients from the
writing style of the accompanying clinical text, which differs between the two
source datasets, and was largely ignoring the image. This form of shortcut learning
is a known risk in medical imaging models. All subsequent models were therefore
evaluated with both standard and misleading prompts.

**Accuracy alone was insufficient.** A more aggressively trained model reached 96%
accuracy with standard prompts but fell to 44% with misleading prompts. A single
accuracy figure would not have revealed this.

**Training data was restructured around the image.** The construction of training
examples was revised so that the text no longer carried information about the
label. Accuracy with misleading prompts increased from 44% to 85%, and accuracy with
a neutral prompt reached 89%.

**Combined robustness techniques degraded the output.** Applying label smoothing
and adversarial examples together caused the model to abandon the required answer
format and produce hedged, conversational responses. Subsequent experiments changed
one factor at a time and kept the training target deterministic.

**Small gains were weighed against robustness.** A later variant improved accuracy
when the prompt contained supportive clinical detail, but detected fewer fibrosis
cases with a neutral prompt (78% compared with 91%) and was less robust. The
difference in standard accuracy amounted to 2 of 77 patients and was not
statistically significant. The more robust model was selected for release.

## Data pipeline

| Step | Description | Rationale |
|---|---|---|
| Slice ordering | Slices are sorted by physical position (`ImagePositionPatient`) rather than file order | File order varies between scanners and cannot be relied on for anatomical position |
| Lower-lobe selection | Slices are taken from the lower 15 to 40% of the lung | Idiopathic pulmonary fibrosis typically has basal predominance. This change gave the largest single improvement in data quality |
| Lung window | -1400 to 240 HU | Displays lung parenchyma, airways and fibrotic patterns such as reticulation and honeycombing |
| Tissue filter | Slices with less than 25% lung tissue are excluded | Removes slices dominated by the diaphragm or air |
| Nodule exclusion | Normal-class slices within 20 mm of an annotated nodule are excluded | Ensures that normal-class images contain normal lung |
| Label noise reduction | Fibrosis-labelled training slices that the base model judged normal were removed before training. Test data was not filtered | A patient with fibrosis does not show it on every lower-lobe slice. Such slices would otherwise teach the model the wrong label |
| Patient-level split | 64/16/20, stratified by class, with no patient in more than one set | Prevents evaluation on patients seen during training |
| Blinded test files | Test images are renamed with random identifiers | Prevents file names from carrying label information |

## Limitations

- **Source confounding.** All fibrosis scans come from OSIC and all normal scans
  from LUNA16. The robustness tests reduce, but do not eliminate, the possibility
  that the model partly recognises differences in scanner or institution. Validation
  on fibrosis and normal scans from a common source is the most important next step.
- **Small test set.** With 81 patients, differences of a few percentage points are
  within statistical noise.
- **No external validation.** All test patients come from the same two datasets
  used for training.
- **Two-dimensional input.** The model assesses individual slices, whereas clinical
  review considers the full volume.
- **Uncalibrated confidence.** The model has produced incorrect predictions with
  confidence above 97%. Its confidence score should not be interpreted as a
  probability.
- **CT only.** On 3,372 NIH chest radiographs the model reached approximately 64%
  accuracy, close to the base model. Features learned from CT do not transfer to
  plain radiographs.

## Repository layout

```
preprocessing/   CT loading, slice ordering, lung windowing, lower-lobe selection, splits
training/        Prompt format, slice dataset and LoRA fine-tuning script
inference/       Model loading and single-slice classification
evaluation/      Patient-level evaluation with 7-slice majority voting
app/             Gradio demo
tests/           Unit tests (no GPU or dataset required)
```

## Quick start

Requirements: Python 3.10, an NVIDIA GPU with at least 8 GB of memory, and a
Hugging Face account that has accepted the
[MedGemma terms](https://huggingface.co/google/medgemma-1.5-4b-it).

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
huggingface-cli login
```

**Run the demo.** The base model and the adapter are downloaded on first use.

```bash
python -m app.gradio_app
```

**Prepare the data.** Download OSIC and LUNA16 from their sources, then:

```bash
python -m preprocessing.build_dataset --osic-dir path/to/osic/train \
    --luna-dir path/to/luna16 --luna-annotations path/to/luna16/annotations.csv \
    --output-dir data/processed
python -m preprocessing.make_splits --manifest data/processed/manifest.json --output-dir data/splits
```

**Train and evaluate.**

```bash
python -m training.train --manifest data/processed/manifest.json --splits-dir data/splits
python -m evaluation.evaluate --manifest data/processed/manifest.json --splits-dir data/splits \
    --adapter models/medgemma-ct-fibrosis
```

`evaluation.evaluate --base` evaluates MedGemma without the adapter for comparison.

**Run the tests.**

```bash
pytest
```

### Notes

- The training script is a reference implementation of the released configuration.
  The label noise reduction step described above is not included, so results may
  differ from the published numbers.
- Patient splits depend on the order of the processed data, so a fresh split will
  not contain exactly the same patients as the published evaluation.
- If downloads from Hugging Face fail behind a corporate proxy or antivirus HTTPS
  inspection, download the base model once and pass its local folder with
  `--base-model`.

## Licences and data

- **Code** in this repository is released under the Apache License 2.0. See
  [`LICENSE`](LICENSE).
- **Model weights** are a derivative of MedGemma and are distributed under the
  [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).
  See [`NOTICE`](NOTICE).
- **Training data is not redistributed.**
  - [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression),
    Open Source Imaging Consortium, CC BY-SA 4.0 under the competition rules. These
    terms prohibit use of the data to cause harm, to discriminate against patients
    with fibrosis, or to re-identify patients.
  - [LUNA16](https://luna16.grand-challenge.org/), CC BY 4.0, derived from
    LIDC-IDRI (CC BY 3.0).

## Acknowledgements

We thank Google Health AI for MedGemma, the Unsloth team, the Open Source Imaging
Consortium, and the LUNA16 and LIDC-IDRI teams for making their models and data
publicly available.
