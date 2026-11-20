"""Versioned prompt files in prompts/<id>.md.

Format:
    ---
    id: translate_v1
    version: 1
    role: translator
    json: true
    ---
    === system ===
    You translate ...
    === user ===
    Translate into {{language}}:
    {{text}}

Variables use {{double braces}} so single-brace placeholders like {PERSON_1} pass
through untouched. Never edit a prompt that has been used for data: copy it to a
new id (translate_v2), because records refer to prompts by id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..common.paths import REPO_ROOT
from .base import LLMRequest, Message

PROMPTS_DIR = REPO_ROOT / "prompts"
_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_SECTION = re.compile(r"^===\s*(system|user)\s*===\s*$", re.M)


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    role: str | None
    json: bool
    system: str
    user: str

    @property
    def variables(self) -> set[str]:
        return set(_VAR.findall(self.system)) | set(_VAR.findall(self.user))

    def _fill(self, template: str, values: dict) -> str:
        def sub(m: re.Match) -> str:
            return str(values[m.group(1)])
        return _VAR.sub(sub, template)

    def render(self, **values) -> list[Message]:
        missing = self.variables - set(values)
        if missing:
            raise PromptError(f"prompt {self.id} is missing variables: {sorted(missing)}")
        msgs = []
        if self.system:
            msgs.append(Message("system", self._fill(self.system, values)))
        msgs.append(Message("user", self._fill(self.user, values)))
        return msgs

    def request(self, **values) -> LLMRequest:
        return LLMRequest(messages=self.render(**values), json_mode=self.json, prompt_id=self.id)


def parse_prompt(text: str, source: str = "<string>") -> Prompt:
    if not text.startswith("---"):
        raise PromptError(f"{source}: missing front matter")
    try:
        _, front, body = text.split("---", 2)
    except ValueError as e:
        raise PromptError(f"{source}: front matter is not closed with ---") from e
    meta = {}
    for line in front.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    if "id" not in meta:
        raise PromptError(f"{source}: front matter needs an id")
    parts = _SECTION.split(body)
    sections = {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts) - 1, 2)}
    if "user" not in sections:
        raise PromptError(f"{source}: needs a '=== user ===' section")
    return Prompt(id=meta["id"], version=meta.get("version", "1"), role=meta.get("role"),
                  json=meta.get("json", "false").lower() == "true",
                  system=sections.get("system", "").strip(), user=sections["user"].strip())


def load_prompt(prompt_id: str, directory: str | Path | None = None) -> Prompt:
    path = Path(directory or PROMPTS_DIR) / f"{prompt_id}.md"
    if not path.exists():
        raise PromptError(f"no prompt file {path}")
    p = parse_prompt(path.read_text(encoding="utf-8"), str(path))
    if p.id != prompt_id:
        raise PromptError(f"{path}: id {p.id!r} does not match file name")
    return p
