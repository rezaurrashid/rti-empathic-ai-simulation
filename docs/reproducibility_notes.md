# Reproducibility notes

## Closed-source LLMs

The study used OpenAI hosted models. Model behavior may change across provider-side updates,
and the provider's full training data, architecture, and training cutoff are not available from the
research code itself.

## Sampling and stochasticity

The simulation cohort is sampled across five factors: age, race, ICD category, social vulnerability
level, and dose per treatment (cGy). Age and dose per treatment are quantile-binned, then combined
with the three categorical variables to define joint strata. Sampling uses `random_state=142` for
reproducible cohort selection.

Turn-specific patient and navigator instructions are selected with Python's `random` module. The
generator exposes `--seed` when deterministic turn-level behavior is desired.

## API inference settings

The pipeline passes model identifiers and messages but does not explicitly pass temperature,
top-p, frequency penalty, presence penalty, or seed. Provider defaults therefore applied.

## Classification caches

Problem labeling and recognition/handling stages used versioned local JSON caches. Caches are
excluded from the public repository because they contain generated text and are implementation
artifacts. Re-running without caches will invoke the configured LLM again.

## Software environment

The public release documents the supported Python dependencies required to run the cleaned
pipeline. Provider-side model behavior may still change over time for hosted LLMs.

## Data-dependent reproduction

The code is public, but exact end-to-end reproduction of the study corpus requires the synthetic
input records and generated conversation files. Those record-level artifacts are withheld from
this public release because record-level release permissions have not been established.

## Human evaluation source

The Google Forms/Sheets response export contains 174 submitted evaluation forms. After
normalizing conversation identifiers such as `248` and `conv_248`, these forms cover 173 unique
conversation IDs; conversation 248 was independently evaluated twice. Clinical appropriateness
items contain 114 responses. The final recommendation item also contains 114 responses in the raw
export. The public repository includes code and aggregate tables but not the raw free-text responses.

The survey export contains no evaluator identifier, so unique-evaluator counts and unique-evaluator
role distributions cannot be independently reconstructed from this CSV alone. Any manuscript claim
about the number or composition of unique evaluators must therefore be supported by the study roster
or another source document.
