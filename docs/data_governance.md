# Data governance

## Approvals

The study underlying this repository was approved by the following review boards:

- Chonnam National University Hospital (CNUH): IRB CNUH-2025-041 & 049; DRB CNUH-D-2025-8
- Wonkwang University Hospital (WKU): WKUH IRB 2025-03-040-003

Informed consent was obtained from all participants during routine clinical care, and records were de-identified prior to research use.

## What is and is not redistributed

| Asset | Status | Notes |
|---|---|---|
| KTAS guideline lookup tables (`guidelines/*.json`) | Released | Adult and pediatric pathways; deterministic `(category, subcategory, modifier) -> level` mapping. |
| Prompt templates (`prompts/*.txt`) | Released | Korean-language strings the pipeline formats with the user's clinical record. |
| Raw clinical notes from CNUH and WKU cohorts | **Not released** | Institutional data-governance restrictions and informed-consent terms do not permit redistribution. |
| Synthetic or sample clinical records | **Not released** | Even fabricated records that mimic the input schema are intentionally excluded from this repository to keep the publication boundary unambiguous. |
| Per-case prediction dumps from the paper experiments | **Not released** | These files reference patient encounters and may contain residual identifying signal. |
| Trained model checkpoints | Available upon request | Derivative artifacts can be shared under a formal Data Use Agreement. |

## Requesting access

Researchers with a need for derivative artifacts (the de-identified evaluation pipeline against the paper cohorts, or trained LoRA adapters) should contact the corresponding author with:

1. A short description of the intended use.
2. Institutional affiliation and a contact at the institution's research-governance office.
3. Willingness to enter a Data Use Agreement (DUA) consistent with CNUH and WKU policy.

The corresponding author is Uichin Lee (uclee@kaist.ac.kr).

## Required CSV schema for new cohorts

The pipeline accepts any dataset that conforms to the column schema below. CSV files should be stored **outside this repository** and passed to the scripts via CLI flags (`--train_data`, `--test_data`, `--stage1_predictions`, etc.).

| Column | Type | Description |
|---|---|---|
| `text` | str | Structured prompt built from the patient record: instruction header + demographics + vital signs + visit info + past medical history + chief complaint + present-illness narrative, terminated by the delimiter `[KTAS sequence]`. |
| `fullseverity` | str | Documented KTAS adjudication sequence as a single comma-separated string: `<age group>, <category>, <subcategory>, <modifier>, <level>`. Mixed Korean / English is permitted. |
| `Initial_Triage_Classification` | int | Documented institutional KTAS level (1-5). Used as the reference target for evaluation and as the ordinal classification label. |
| `Research_ID`, `Extracted_ID` | str | Optional identifiers used to join predictions across runs. |

To use the pipeline on a new cohort:

1. Ensure your institution's data-governance and consent terms permit the planned analysis.
2. Convert your records to the schema above.
3. If your guideline differs from KTAS, replace the JSON lookups in `guidelines/`; the candidate-retrieval logic indexes them by category and subcategory.

The code makes no calls to external services other than (a) the Hugging Face Hub for downloading the open-weights backbone, and (b) the OpenAI API when the GPT-4o baseline is explicitly enabled.
