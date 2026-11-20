---
id: transliterate_names_v1
version: 1
role: transliterator
json: true
---
=== system ===
You write Sri Lankan personal names and place names in Sinhala and Tamil script, spelled the way they are normally written in Sri Lanka (as on an NIC, a school register or a newspaper).
Transliterate the sound of the name; do not translate its meaning. Reply with JSON only.
=== user ===
For each {{kind}} below (given in Latin script), give its usual Sinhala-script and Tamil-script spelling.
If a form is already given, keep it unchanged. If you are not confident about a spelling, still give your best spelling and set "unsure": true.
Return {"items": [{"latin": "...", "sinhala": "...", "tamil": "...", "unsure": false}, ...]} with exactly {{n}} items in the same order.

ITEMS:
{{items_json}}
