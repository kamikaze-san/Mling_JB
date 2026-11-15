"""
GGUF target runner (llama.cpp) — the deployed artifact we evaluate.

Every safety / capability number in this project is measured on a GGUF file
(official CohereLabs tiny-aya-global Q4_K_M by default), optionally with a
GGUF LoRA adapter applied at runtime. The base file stays byte-identical
across conditions; only the adapter changes.

Design points:
  - fixed decoding (temperature 0 + fixed seed by default) for every checkpoint
  - multi-turn chat via the GGUF's embedded chat template
  - persistent disk cache keyed on (base fingerprint, adapter fingerprint,
    decoding settings, full message list) -> resumable, zero repeated compute
  - llama_cpp is imported lazily so the module is importable in tests
"""

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional

from src.harness.cache import GenerationCache


def file_fingerprint(path: Optional[str]) -> str:
    """Cheap, stable fingerprint: basename + size + sha256 of first/last 1 MiB."""
    if not path:
        return "none"
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return f"{os.path.basename(path)}:{st.st_size}:{h.hexdigest()[:16]}"


class GGUFRunner:
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        seed: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        cache: Optional[GenerationCache] = None,
        llama_factory: Optional[Callable[..., Any]] = None,
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.cache = cache or GenerationCache()
        self._llama_factory = llama_factory
        self._llm = None
        self.model_id = (
            f"gguf[{file_fingerprint(model_path)}]"
            f"+lora[{file_fingerprint(lora_path)}@{lora_scale}]"
        )

    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """Decoding/config record for run manifests."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "lora_path": self.lora_path,
            "lora_scale": self.lora_scale,
            "n_ctx": self.n_ctx,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
        }

    def _load(self):
        if self._llm is not None:
            return
        factory = self._llama_factory
        if factory is None:
            from llama_cpp import Llama  # lazy: heavy native dependency
            factory = Llama
        kwargs = dict(model_path=self.model_path, n_ctx=self.n_ctx,
                      n_gpu_layers=self.n_gpu_layers, seed=self.seed, verbose=False)
        if self.lora_path:
            kwargs["lora_path"] = self.lora_path
            kwargs["lora_scale"] = self.lora_scale
        self._llm = factory(**kwargs)

    def _messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for m in messages:
            content = m["content"]
            if isinstance(content, list):  # tolerate multimodal-style content lists
                content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
            msgs.append({"role": m["role"], "content": content})
        return msgs

    def chat(self, messages: List[Dict[str, str]], tags: Optional[Dict[str, Any]] = None,
             skip_cache: bool = False) -> str:
        """Generate the next assistant turn for a full message list (user/assistant alternating)."""
        msgs = self._messages(messages)
        prompt_key = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        cache_model_id = f"{self.model_id}|seed={self.seed}|top_p={self.top_p}"
        if not skip_cache:
            hit = self.cache.get(model_id=cache_model_id, prompt=prompt_key,
                                 temperature=self.temperature, max_new_tokens=self.max_tokens)
            if hit is not None:
                return hit
        self._load()
        out = self._llm.create_chat_completion(
            messages=msgs,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        text = (out["choices"][0]["message"]["content"] or "").strip()
        self.cache.set(model_id=cache_model_id, prompt=prompt_key, response=text,
                       temperature=self.temperature, max_new_tokens=self.max_tokens, tags=tags)
        return text

    def generate(self, prompt: str, tags: Optional[Dict[str, Any]] = None) -> str:
        """Single-turn convenience wrapper."""
        return self.chat([{"role": "user", "content": prompt}], tags=tags)

    def replay(self, user_turns: List[str], tags: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Replay a fixed sequence of user turns, feeding back the model's own
        responses. Returns one response per user turn.
        """
        history: List[Dict[str, str]] = []
        responses = []
        for i, u in enumerate(user_turns):
            history.append({"role": "user", "content": u})
            r = self.chat(history, tags={**(tags or {}), "turn": i + 1})
            history.append({"role": "assistant", "content": r})
            responses.append(r)
        return responses
