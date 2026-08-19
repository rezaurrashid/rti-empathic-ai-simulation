# Data availability

The code was developed around a synthetic tabular radiation-therapy dataset generated from a
de-identified institutional cohort.

This public release does **not** contain:

- the source institutional clinical dataset;
- the full 8,205-row synthetic tabular dataset;
- the 300 generated persona cards;
- the 300 full conversation transcripts;
- OpenAI response caches;
- problem-level CSVs containing patient/assistant text;
- the raw human-evaluation survey export and free-text evaluator comments.

This conservative default prevents accidental public release of study-derived record-level data
before the authors confirm institutional and IRB/data-governance requirements.

## What is included

- `synthetic_input_schema.csv`: expected tabular schema.
- `../examples/example_synthetic_records.csv`: fabricated format examples, not study records.
- `../results/aggregate/`: aggregate non-text source data supporting the principal quantitative results.

If the study team later confirms that the synthetic tabular data and/or generated conversations
may be released publicly, place them in a versioned data archive and document the exact release
in the manuscript Data Availability statement.

## Human evaluation data

The survey analysis can be reproduced locally with `scripts/06_summarize_human_evaluation.py` if
the authorized raw Google Forms/Sheets export is available. Only aggregate outputs are included in
this public release. Public release of the raw survey responses should be considered
separately under the consent language and institutional data-sharing requirements.
