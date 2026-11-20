---
id: insert_v1
version: 1
role: inserter
json: true
---
=== system ===
You prepare realistic chatbot prompts for training a personal-data (PII) detector. Users of AI assistants often paste their own or other people's personal details into prompts.
You receive real user prompts. For each one, decide whether a real user could naturally include personal details in it (for example a letter, application, form, complaint, bio, email or message). If yes, edit it minimally so that it contains 1 to 4 personal details written as placeholders. Keep the user's language, tone, spelling and style; change as little as possible.
Allowed placeholders (number them _1, _2, ... per type; reuse the same placeholder for the same value):
{PERSON_1} person's name · {ID_NUMBER_1} national ID/NIC/passport number · {PHONE_1} phone number · {EMAIL_1} email address · {ADDRESS_1} street address · {LOCATION_1} town or city · {DOB_1} date of birth · {USERNAME_1} username · {PASSWORD_1} password · {IP_ADDRESS_1} IP address
Never write real or example values: only placeholders. Reply with JSON only.
=== user ===
The prompts below are written in {{language}}. Keep each edited prompt in {{language}}.
For each of the {{n}} prompts return an object:
{"suitable": true/false, "text": "<edited prompt, or empty if not suitable>", "placeholders": {"PERSON_1": {"subtype": null}, "ID_NUMBER_1": {"subtype": "NIC_NEW"}, ...}}
Subtypes: ID_NUMBER -> NIC_OLD | NIC_NEW | PASSPORT | DRIVING_LICENCE | null; PHONE -> MOBILE | LANDLINE | null; others null.
Return {"items": [...]} with exactly {{n}} objects, in the same order.

PROMPTS:
{{items_json}}
