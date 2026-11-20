---
id: smoke_v1
version: 1
role: smoke
json: true
---
=== system ===
You are a connectivity test. Reply with JSON only.
=== user ===
Translate the English word "{{word}}" into Sinhala and Tamil script, keeping the token {PERSON_1} unchanged.
Reply as JSON: {"si": "<Sinhala> {PERSON_1}", "ta": "<Tamil> {PERSON_1}"}
