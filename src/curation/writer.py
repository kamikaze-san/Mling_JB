"""
Response writer for preference data, via any OpenAI-compatible chat endpoint
(LM Studio / oMLX / mlx_lm.server serving the local Qwen model).

It writes only two kinds of text:
  - a brief, non-preachy refusal (the "chosen" reply to an attack turn, and the
    synthetic false refusal used as "rejected" for a benign twin)
  - a helpful answer to a benign prompt (only when the base model itself
    over-refused, so no on-policy helpful answer exists)
Both are written in the same language / script / register as the user's
latest message. Results are cached on disk.
"""

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from src.harness.cache import GenerationCache

REFUSAL_SYSTEM = (
    "You write the assistant's reply for a safety training dataset. The user's latest "
    "message asks for something the assistant must not help with. Write a short reply "
    "(one to three sentences) that clearly declines without lecturing, moralising, or "
    "repeating any harmful detail. Reply in exactly the same language, script and register "
    "as the user's latest message, including any mixing of languages. Where natural, offer "
    "a safe, related alternative. Output only the reply."
)

HELPFUL_SYSTEM = (
    "You write the assistant's reply for a training dataset. Answer the user's latest "
    "message helpfully, accurately and completely. Reply in exactly the same language, "
    "script and register as the user's message, including any mixing of languages. "
    "Output only the reply."
)

FALSE_REFUSAL_SYSTEM = (
    "You write an example of an unhelpful over-cautious assistant reply for a training "
    "dataset (it will be used as the dispreferred answer). Write one or two sentences that "
    "politely decline the user's harmless request as if it were unsafe. Reply in exactly "
    "the same language, script and register as the user's message. Output only the reply."
)


class ResponseWriter:
    def __init__(self, base_url: str, model: str, api_key: str = "local",
                 temperature: float = 0.3, max_tokens: int = 512,
                 cache: Optional[GenerationCache] = None, timeout: int = 300,
                 disable_thinking: bool = True):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache or GenerationCache()
        self.timeout = timeout
        self.disable_thinking = disable_thinking

    def _call(self, system: str, messages: List[Dict[str, str]]) -> str:
        msgs = [{"role": "system", "content": system}] + messages
        key = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        hit = self.cache.get(model_id=f"writer:{self.model}", prompt=key,
                             temperature=self.temperature, max_new_tokens=self.max_tokens)
        if hit is not None and hit.strip():
            return hit
        body = {"model": self.model, "messages": msgs, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["reasoning_effort"] = "none"
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code == 400 and "reasoning_effort" in body:
                body.pop("reasoning_effort")
                req = urllib.request.Request(
                    f"{self.base_url}/v1/chat/completions", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = json.loads(r.read())["choices"][0]["message"]["content"] or ""
            else:
                raise
        text = strip_think(text)
        if text.strip():
            self.cache.set(model_id=f"writer:{self.model}", prompt=key, response=text,
                           temperature=self.temperature, max_new_tokens=self.max_tokens)
        return text

    def refusal(self, messages: List[Dict[str, str]]) -> str:
        return self._call(REFUSAL_SYSTEM, messages)

    def helpful(self, messages: List[Dict[str, str]]) -> str:
        return self._call(HELPFUL_SYSTEM, messages)

    def false_refusal(self, messages: List[Dict[str, str]]) -> str:
        return self._call(FALSE_REFUSAL_SYSTEM, messages)


def strip_think(text: str) -> str:
    """Remove a leading <think>...</think> block if the server returned one inline."""
    t = text or ""
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    return t.strip()
