# Methods-to-code mapping

| Study component | Public script/function |
|---|---|
| Synthetic-record sampling | `scripts/01_generate_simulations.py::stratified_sample` (five-factor stratified sampling) |
| Persona generation | `scripts/01_generate_simulations.py::build_persona_prompt` and `generate_persona_card` |
| Navigator prompt | `scripts/01_generate_simulations.py::assistant_prompt` |
| Patient-agent prompt | `scripts/01_generate_simulations.py::patient_prompt` |
| Dialogue termination | `scripts/01_generate_simulations.py::should_end_conversation` |
| Reference-free dialogue metrics | `scripts/02_compute_automated_metrics.py` |
| 19-category problem taxonomy | `scripts/03_classify_patient_problems.py::PROBLEM_CATEGORIES` |
| Problem-specific severity | `scripts/03_classify_patient_problems.py::build_classification_prompt` |
| Recognition/handling definitions | `scripts/04_evaluate_recognition_handling.py::build_recognition_prompt` |
| Hierarchical recognition post-processing | `scripts/04_evaluate_recognition_handling.py::evaluate_assistant_response` |
| Severity-stratified performance | `scripts/05_build_severity_tables.py` |
| Conversation-level bootstrap CI | `scripts/05_build_severity_tables.py::bootstrap_ci` |
| Human evaluation aggregation | `scripts/06_summarize_human_evaluation.py` |
| Core manuscript plots | `analysis/generate_manuscript_outputs.py` |
