import os
import logging
import torch
from typing import Optional, Dict, Any
from src.harness.device import resolve_device, get_quantization_config
from src.harness.cache import GenerationCache

logger = logging.getLogger(__name__)

class ModelRunner:
    """
    Device-agnostic inference runner for Cohere Aya and evaluation models.
    Supports:
    - 4-bit NF4 Quantization (RTX 4060 8GB VRAM)
    - MPS (Apple Silicon Mac Mini M4 48GB)
    - CUDA standard (Colab / Kaggle)
    - Automatic persistent disk caching (SQLite + JSONL)
    """
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_4bit: bool = True,
        cache: Optional[GenerationCache] = None,
        hf_token: Optional[str] = None
    ):
        self.model_id = model_id
        self.device, self.dtype = resolve_device(device)
        self.load_in_4bit = load_in_4bit
        self.cache = cache or GenerationCache()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self):
        """Lazy loader for model weights and tokenizer."""
        if self._is_loaded:
            return

        logger.info(f"Loading {self.model_id} on {self.device} (4-bit={self.load_in_4bit})...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
            trust_remote_code=True
        )

        quant_config = get_quantization_config(self.device, self.load_in_4bit)

        load_kwargs: Dict[str, Any] = {
            "token": self.hf_token,
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }

        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        elif self.device == "mps":
            load_kwargs["device_map"] = {"": "mps"}

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._is_loaded = True
        logger.info(f"Successfully loaded {self.model_id}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.6,
        max_new_tokens: int = 512,
        tags: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False
    ) -> str:
        """
        Generate text with persistent caching.
        If prompt was previously computed, returns immediately from disk cache.
        """
        # 1. Check disk cache
        if not skip_cache:
            cached_res = self.cache.get(
                model_id=self.model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            if cached_res is not None:
                return cached_res

        # 2. Lazy load weights if not already in memory
        if not self._is_loaded:
            self.load()

        # 3. Format message with tokenizer chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_input = f"{system_prompt}\nUser: {prompt}\nAssistant:"

        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        target_device = "cuda" if self.device == "cuda" else self.device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_tokens[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 4. Save to persistent cache
        self.cache.set(
            model_id=self.model_id,
            prompt=prompt,
            response=response,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tags=tags
        )

        return response

