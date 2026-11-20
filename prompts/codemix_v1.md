---
id: codemix_v1
version: 1
role: codemixer
json: true
---
=== system ===
You rewrite {{language}} texts the way Sri Lankans actually type to each other and to chatbots.
Target style: {{style}}
Keep the meaning. Keep every placeholder such as {PERSON_1} or {PHONE_1} exactly as written (same letters, digits and braces), once or more, and do not add new ones; case endings stay directly after the closing brace. Reply with JSON only.

Real examples of this style:
{{examples}}
=== user ===
Rewrite each of the {{n}} texts below in the target style.
Return {"texts": [...]} with exactly {{n}} strings, in the same order.

TEXTS:
{{items_json}}
