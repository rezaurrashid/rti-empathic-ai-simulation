# Prompt Library

This directory documents the prompts used in the final analysis pipeline.

The **executable source of truth** is the prompt-builder functions in `scripts/`:

- `scripts/01_generate_simulations.py`
  - `build_persona_prompt`
  - `assistant_prompt`
  - `patient_prompt`
- `scripts/03_classify_patient_problems.py`
  - `build_classification_prompt`
- `scripts/04_evaluate_recognition_handling.py`
  - `build_recognition_prompt`

The generation prompts are dynamic. Conversation history, persona information, turn number,
open-ended-question count, and randomized behavioral instructions are inserted at runtime.
For this reason, the files in this folder describe the stable prompt text and the dynamic fields,
while the Python functions preserve the exact executable logic.

No hidden prompt files are required to run the released code.
