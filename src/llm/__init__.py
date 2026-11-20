"""LLM client layer: Gemini, OpenRouter, DeepSeek (and an offline mock).

    from src.llm import get_llm, load_prompt
    llm = get_llm("translator")                # role from configs/llm.toml
    req = load_prompt("smoke_v1").request(word="name")
    data, resp = llm.complete_json(req)
    record["generation"]["models"].append(resp.as_model_record("translator"))
"""

from .base import LLMError, LLMRequest, LLMResponse, Message
from .batch import BatchResult, run_batch
from .client import LLM, get_llm
from .config import load_config
from .jsonout import JSONReplyError, parse_json
from .prompts import Prompt, load_prompt

__all__ = ["LLM", "LLMError", "LLMRequest", "LLMResponse", "Message", "BatchResult", "run_batch",
           "get_llm", "load_config", "parse_json", "JSONReplyError", "Prompt", "load_prompt"]
