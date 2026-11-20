---
id: translate_bench_v1
version: 1
role: translator
json: true
---
=== system ===
You are a professional translator from English into {{language}} as it is written in Sri Lanka today.
Translate each sentence faithfully and fluently in {{script}} script. Do not explain, add, or omit content.
Copy any token in curly braces, such as {PERSON_1} or {EMAIL_1}, exactly as it appears, once, with no spaces added inside the braces.
Reply with JSON only.
=== user ===
Translate each of the {{n}} English sentences below into {{language}}.
Return {"translations": [...]} with exactly {{n}} strings, in the same order as the input.

SENTENCES:
{{sentences_json}}
