# Persona Generation Prompt

**Model default in the production notebook:** `gpt-5`

The following stable text is used by `build_persona_prompt()` in
`scripts/01_generate_simulations.py`. Values in braces are populated from one sampled
synthetic tabular record.

```text
Generate a realistic oncology patient persona going through radiation therapy.

IMPORTANT: Reflect the patient's actual race, ethnicity, and cultural background authentically.
Include cultural values, family structures, communication styles, and community resources specific to their background.

Create a patient who can be:
- Angry, frustrated, or irritable (cancer is hard!)
- Skeptical of help or resistant to suggestions
- Emotionally guarded or slow to trust
- Occasionally difficult or demanding
- Experiencing depression, anxiety, or hopelessness

Include: age, race, gender, cancer diagnosis, emotional state (be honest - can be angry/depressed),
job stability, education level, health literacy, technology comfort, family support,
financial situation, lifestyle habits, and communication style and complexity.
Specify if they have trouble reading, following instructions, or using technology.
Make realistic and diverse (10-12 sentences).

Demographics: Age {AGE}, {GENDER}, {RACE}, {MARITAL_STATUS}.
Cancer: ICD-10 {ICD_category} (dose: {TX_PLN1_PRSCRB_DOSE_CGY} cGy).
Missed sessions: {missing_days_C1}. Distance: {distance_to_rad_facility_in_mile} miles.
Income: ${medianhouseholdincome}, Insurance: {INSURANCE}.
Social vulnerability: {socialvulnerabilitylevel}.
Lifestyle: Smoking {smoking_status}, Alcohol {alcohol_use}.
```

The prompt does **not** directly insert education, health-literacy, technology-comfort,
family-support, or job-stability fields from the tabular record. Those attributes were requested
from the model in the narrative-generation instruction.
