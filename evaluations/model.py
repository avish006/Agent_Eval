import os
import re
import json
import time
import asyncio
from typing import Optional, Dict, List, Tuple, Union
from dotenv import load_dotenv
from pydantic import BaseModel
from deepeval.models.llms.ollama_model import OllamaModel

load_dotenv()

model_name = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Regex to strip markdown code fences that gemma4 sometimes wraps around JSON
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences (```json ... ```) wrapping JSON output."""
    m = _CODE_FENCE_RE.match(text)
    return m.group(1).strip() if m else text.strip()


def _try_parse_json(text: str, schema: BaseModel):
    """Try to parse text as JSON conforming to the given schema.
    Strips code fences first, then attempts validation.
    Returns (parsed_object, None) on success or (None, error_string) on failure.
    """
    cleaned = _strip_code_fence(text)
    try:
        return schema.model_validate_json(cleaned), None
    except Exception as e:
        return None, str(e)


class ResilientOllamaModel(OllamaModel):
    """
    Subclass of DeepEval's OllamaModel that adds:
    1. Strips markdown code fences from structured (schema) responses
    2. Semaphore-based concurrency limiting (avoids flooding Ollama Cloud)
    3. Exponential backoff and retry for 429 'too many concurrent requests'
    4. Re-prompt fallback when JSON parsing fails
    5. Safe fallbacks for model_data being None (cloud models not in registry)
    """
    _async_semaphore: Optional[asyncio.Semaphore] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = 8
        self.base_delay = 2.0

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._async_semaphore is None:
            cls._async_semaphore = asyncio.Semaphore(2)
        return cls._async_semaphore

    # ---- Capability overrides (model_data fields may be None for cloud models) ----
    def supports_log_probs(self) -> Union[bool, None]:
        if self.model_data and self.model_data.supports_log_probs is not None:
            return self.model_data.supports_log_probs
        return False

    def supports_temperature(self) -> Union[bool, None]:
        if self.model_data and self.model_data.supports_temperature is not None:
            return self.model_data.supports_temperature
        return True

    def supports_multimodal(self) -> Union[bool, None]:
        if self.model_data and self.model_data.supports_multimodal is not None:
            return self.model_data.supports_multimodal
        return False

    def supports_structured_outputs(self) -> Union[bool, None]:
        if self.model_data and self.model_data.supports_structured_outputs is not None:
            return self.model_data.supports_structured_outputs
        return True

    def supports_json_mode(self) -> Union[bool, None]:
        if self.model_data and getattr(self.model_data, 'supports_json', None) is not None:
            return self.model_data.supports_json
        return True

    # ---- Helper: call Ollama chat and parse response ----
    def _do_chat(self, prompt, schema, audio):
        chat_model = self.load_model()
        messages = self._prompt_messages(prompt, audio)
        response = chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
        )
        return response.message.content

    async def _a_do_chat(self, prompt, schema, audio):
        chat_model = self.load_model(async_mode=True)
        messages = self._prompt_messages(prompt, audio)
        response = await chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
        )
        return response.message.content

    # ---- Core generate with code-fence stripping + retry ----
    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List] = None,
    ) -> Tuple[Union[str, BaseModel], float]:
        for attempt in range(self.max_retries):
            try:
                content = self._do_chat(prompt, schema, audio)

                if not schema:
                    return content, 0

                parsed, err = _try_parse_json(content, schema)
                if parsed is not None:
                    return parsed, 0

                # JSON parse failed — retry with explicit JSON instruction
                json_prompt = (
                    f"{prompt}\n\nYou MUST respond with ONLY a valid JSON object "
                    f"matching this schema (no markdown, no code fences, no extra text):\n"
                    f"{json.dumps(schema.model_json_schema(), indent=2)}"
                )
                content2 = self._do_chat(json_prompt, schema, audio)
                parsed2, err2 = _try_parse_json(content2, schema)
                if parsed2 is not None:
                    return parsed2, 0

                raise ValueError(f"JSON parse failed after re-prompt: {err2}")

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "too many concurrent" in err_str or "rate limit" in err_str:
                    wait_time = self.base_delay * (1.5 ** attempt)
                    print(f"\n[Ollama 429] Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1 and ("json" in err_str or "validation" in err_str):
                    wait_time = self.base_delay * (1.2 ** attempt)
                    print(f"\n[Ollama JSON Error] Retrying in {wait_time:.1f}s ({attempt + 1}/{self.max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise

        # Final attempt
        content = self._do_chat(prompt, schema, audio)
        if schema:
            parsed, _ = _try_parse_json(content, schema)
            if parsed is not None:
                return parsed, 0
            raise ValueError(f"JSON parse failed after all retries. Last response: {content[:200]}")
        return content, 0

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List] = None,
    ) -> Tuple[Union[str, BaseModel], float]:
        sem = self._get_semaphore()
        async with sem:
            for attempt in range(self.max_retries):
                try:
                    content = await self._a_do_chat(prompt, schema, audio)

                    if not schema:
                        return content, 0

                    parsed, err = _try_parse_json(content, schema)
                    if parsed is not None:
                        return parsed, 0

                    # JSON parse failed — retry with explicit JSON instruction
                    json_prompt = (
                        f"{prompt}\n\nYou MUST respond with ONLY a valid JSON object "
                        f"matching this schema (no markdown, no code fences, no extra text):\n"
                        f"{json.dumps(schema.model_json_schema(), indent=2)}"
                    )
                    content2 = await self._a_do_chat(json_prompt, schema, audio)
                    parsed2, err2 = _try_parse_json(content2, schema)
                    if parsed2 is not None:
                        return parsed2, 0

                    raise ValueError(f"JSON parse failed after re-prompt: {err2}")

                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "too many concurrent" in err_str or "rate limit" in err_str:
                        wait_time = self.base_delay * (1.5 ** attempt)
                        print(f"\n[Ollama 429] Waiting {wait_time:.1f}s before async retry {attempt + 1}/{self.max_retries}...")
                        await asyncio.sleep(wait_time)
                    elif attempt < self.max_retries - 1 and ("json" in err_str or "validation" in err_str):
                        wait_time = self.base_delay * (1.2 ** attempt)
                        print(f"\n[Ollama JSON Error] Async retry in {wait_time:.1f}s ({attempt + 1}/{self.max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            # Final attempt
            content = await self._a_do_chat(prompt, schema, audio)
            if schema:
                parsed, _ = _try_parse_json(content, schema)
                if parsed is not None:
                    return parsed, 0
                raise ValueError(f"JSON parse failed after all retries. Last response: {content[:200]}")
            return content, 0


eval_model = ResilientOllamaModel(model=model_name, base_url=base_url)
