# Stress-Testing Empathic AI Navigation for Radiation Therapy Interruptions

**Release:** v1.0.0  
**License:** MIT  
**Associated manuscript:** *Stress-Testing Empathic AI Navigation for Radiation Therapy Interruptions Using Synthetic Patient Dialogues*

Public research-code release for a dual-agent simulation framework used to generate synthetic
oncology patient personas, simulate multi-turn patient–navigator conversations, and evaluate
problem recognition, actionability, severity-sensitive handling, and dialogue quality.

The repository contains the final simulation and evaluation pipeline, prompt documentation,
aggregate study outputs, and reproducibility notes.

## Study workflow

```text
Synthetic tabular RT records
        |
        v
01_generate_simulations.py
  GPT-5 persona generation
  o3 patient agent
  o3 navigator agent
        |
        v
conv_001.docx ... conv_300.docx
        |
        +------------------------------+
        |                              |
        v                              v
02_compute_automated_metrics.py   03_classify_patient_problems.py
reference-free dialogue metrics   19-category problem taxonomy
                                  problem-specific LOW/MEDIUM/HIGH severity
                                         |
                                         v
                                  04_evaluate_recognition_handling.py
                                  GPT-5 recognition/action assessment
                                         |
                                         v
                                  05_build_severity_tables.py
                                  severity-stratified summaries + bootstrap CIs
        |                              |
        +---------------+--------------+
                        |
                        +-------------------------------+
                        |                               |
                        v                               v
             06_summarize_human_evaluation.py   analysis/generate_manuscript_outputs.py
             aggregate survey summaries         manuscript figures/source tables
```

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── .env.example
├── .gitignore
├── scripts/
│   ├── 01_generate_simulations.py
│   ├── 02_compute_automated_metrics.py
│   ├── 03_classify_patient_problems.py
│   ├── 04_evaluate_recognition_handling.py
│   ├── 05_build_severity_tables.py
│   └── 06_summarize_human_evaluation.py
├── analysis/
│   └── generate_manuscript_outputs.py
├── prompts/
│   ├── README.md
│   ├── persona_generation.md
│   ├── navigator_agent.md
│   ├── patient_agent.md
│   ├── problem_classification.md
│   └── recognition_handling.md
├── data/
│   ├── README.md
│   └── synthetic_input_schema.csv
├── examples/
│   └── example_synthetic_records.csv
├── results/
│   ├── README.md
│   └── aggregate/
└── docs/
    ├── methods_mapping.md
    └── reproducibility_notes.md
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

The dependency file specifies the packages required to run the cleaned public pipeline.

## OpenAI API configuration

Copy `.env.example` to `.env` and add an API key:

```text
OPENAI_API_KEY=your_key_here
```

Do not commit `.env` or any API credential.

## Data

The institutional radiation-therapy dataset is **not included** in this repository.

The generation script expects a synthetic tabular CSV with the schema documented in
`data/synthetic_input_schema.csv`. `examples/example_synthetic_records.csv` contains fabricated
rows for format illustration only and is not study data.

The public release excludes the full study-generated synthetic records,
persona cards, conversation transcripts, model caches, and problem-level text outputs until
their release status is confirmed by the study team and institutional data-governance process.
See `data/README.md`.

## Reproducing the pipeline

### 1. Generate persona cards and conversations

```bash
python scripts/01_generate_simulations.py \
  --input /path/to/synthetic_train_data.csv \
  --output-dir outputs/conversations \
  --n 300
```

Files are written directly as `conv_001.docx` through `conv_300.docx`.

The 300-record simulation cohort is sampled across five factors: **age, race, ICD category,
social vulnerability level, and dose per treatment (cGy)**. Age and dose per treatment are
quantile-binned before joint stratification so that the sampled cohort spans demographic,
clinical, social, and treatment-complexity variation in the synthetic population.

### 2. Compute automated dialogue metrics

```bash
python scripts/02_compute_automated_metrics.py \
  --conversations outputs/conversations \
  --output outputs/evaluation/automated_metrics.csv
```

Metrics include semantic response relevance, lexical diversity, response-length characteristics,
SDoH relevance, empathy markers, question patterns, contextual coherence, emotional consistency,
and action orientation.

### 3. Classify patient-raised problems and severity

```bash
python scripts/03_classify_patient_problems.py \
  --conversations outputs/conversations \
  --output-dir outputs/evaluation
```

Outputs:

- `utterance_labels.csv`
- `problem_instance_labels.csv`

The final taxonomy contains 19 problem categories. Severity is assigned separately to each
problem mention as LOW, MEDIUM, or HIGH.

### 4. Evaluate recognition and handling

```bash
python scripts/04_evaluate_recognition_handling.py \
  --conversations outputs/conversations \
  --problem-labels outputs/evaluation/problem_instance_labels.csv \
  --output-dir outputs/evaluation
```

Outputs:

- `assistant_recognition_rates_problem_specific.csv`
- `problem_instance_recognition.csv`

The post-processing hierarchy is enforced programmatically:

```text
handled ⊆ recognized
missed = all patient-raised problems - recognized
```

### 5. Build severity-stratified summaries

```bash
python scripts/05_build_severity_tables.py \
  --problem-labels outputs/evaluation/problem_instance_labels.csv \
  --recognition outputs/evaluation/problem_instance_recognition.csv \
  --output-dir outputs/evaluation/summary
```

The default confidence intervals use 1,000 conversation-level bootstrap resamples with seed 42.

### 6. Summarize human evaluation responses

The raw Google Forms/Sheets export is treated as restricted study data and is not committed.
Given a local copy of the response CSV:

```bash
python scripts/06_summarize_human_evaluation.py \
  --input /path/to/human_evaluation_responses.csv \
  --output-dir results/aggregate
```

The script writes only aggregate Likert summaries, response-level demographic summaries, and
conversation-coverage counts. It does not export free-text responses. Agreement is defined as a
Likert rating of 6 or 7.

### 7. Generate core manuscript figures

```bash
python analysis/generate_manuscript_outputs.py \
  --automated-metrics outputs/evaluation/automated_metrics.csv \
  --recognition-summary outputs/evaluation/assistant_recognition_rates_problem_specific.csv \
  --problem-labels outputs/evaluation/problem_instance_labels.csv \
  --problem-recognition outputs/evaluation/problem_instance_recognition.csv \
  --output-dir outputs/manuscript
```

## Prompt transparency

All prompt logic is public.

The prompt documentation is under `prompts/`, while the Python prompt-builder functions are the
executable source of truth. This is important because both dialogue agents use dynamic prompt
components based on conversation history, turn number, detected open-ended questions, and
randomized behavioral instructions.

## Model defaults used in the final code path

| Pipeline step | Model |
|---|---|
| Persona generation | `gpt-5` |
| Synthetic patient agent | `o3` |
| AI navigator agent | `o3` |
| Problem category and severity classification | `gpt-5` |
| Recognition and handling assessment | `gpt-5` |

No temperature, top-p, frequency penalty, or seed is explicitly passed to the OpenAI API, so
provider defaults apply. Turn-level behavioral instructions use Python's `random` module; the
generation script provides an optional `--seed` argument when deterministic reruns are desired.

## Aggregate study outputs

`results/aggregate/` contains small, non-text aggregate source tables derived from the latest
problem-specific evaluation files and the human-evaluation survey export. They are included
to make the principal reported quantities auditable without distributing the full conversation corpus
or raw free-text evaluator responses.

## Scope

This code supports **preclinical simulation and evaluation** of candidate conversational AI
navigation systems. It is not a deployed clinical navigator, medical device, diagnostic system,
or substitute for clinician or patient-navigator judgment.

## Citation

Citation metadata are provided in `CITATION.cff`. After the GitHub repository is created, add its
public repository URL to `CITATION.cff` using the `repository-code` field. A manuscript DOI can be
added later when a final bibliographic record is available.

## License

This repository is released under the **MIT License**. Copyright is held by the authors. See
[`LICENSE`](LICENSE) for the full terms.
