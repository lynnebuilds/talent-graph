# -*- coding: utf-8 -*-
"""
High-Performance Asynchronous LLM Client for Official DeepSeek & Multi-Relay OpenAI-Compatible APIs.
Features:
- Official DeepSeek API (default model: deepseek-flash) + arbitrary named relay stations (RELAY_<NAME>_*)
- Built-in Semaphore Rate-Limiter (default MAX_CONCURRENCY=20)
- Exponential Backoff Retry for network glitches and 429 rate limits
- Automatic fallback to Official DeepSeek when a relay is misconfigured or fails after retries
- Pydantic V2 Structured JSON extraction with robust fallback parser
"""

import asyncio
import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    RateLimitError,
    Timeout,
)
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

try:
    from httpx2 import Limits as HttpxLimits
except ImportError:  # pragma: no cover - fallback for openai 1.x environments
    from httpx import Limits as HttpxLimits  # type: ignore[assignment,no-redef]

from configs.settings import Settings, get_settings
from src.core.logger import get_logger

logger = get_logger("llm_client")
T = TypeVar("T", bound=BaseModel)

_URL_REDACT_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_KEY_REDACT_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{4,}\b")


def _sanitize_error_for_log(exc: Exception) -> str:
    """脱敏异常摘要，防止私有中转站 URL 或 API Key 片段泄漏至日志"""
    status = getattr(exc, "status_code", None)
    msg = _URL_REDACT_RE.sub("[REDACTED_URL]", str(exc))
    msg = _KEY_REDACT_RE.sub("[REDACTED_KEY]", msg)
    if len(msg) > 240:
        msg = f"{msg[:240]}..."
    status_part = f" status={status}" if status is not None else ""
    return f"{type(exc).__name__}{status_part}: {msg}"


class LLMClient:
    """高性能异步限流多渠道 LLM 客户端（官方 DeepSeek + 多中转站 + 自动降级）"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider_name: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)

        self.profile, self.fallback_reason = self.settings.resolve_llm_provider(provider_name)
        if self.fallback_reason:
            logger.warning(self.fallback_reason)

        self.provider_name: str = self.profile.name
        self.base_url: str = self.profile.base_url
        self.default_model: str = self.profile.model
        self.is_official: bool = self.profile.is_official

        self._client = self._build_openai_client(
            api_key=self.profile.api_key or "mock_key",
            base_url=self.profile.base_url,
        )

        # 若当前使用的是中转站且官方 DeepSeek 密钥有效，预建官方降级客户端（官方自身永不自递归）
        self._fallback_client: Optional[AsyncOpenAI] = None
        self._fallback_model: Optional[str] = None
        if not self.is_official and self.provider_name != "mock":
            official = self.settings.get_official_deepseek_profile()
            if Settings.is_valid_api_key(official.api_key):
                self._fallback_client = self._build_openai_client(
                    api_key=official.api_key,
                    base_url=official.base_url,
                )
                self._fallback_model = official.model

        self.total_requests = 0
        self.total_success = 0
        self.total_failures = 0
        self.total_fallbacks = 0
        self.total_latency_seconds = 0.0

    def _build_openai_client(self, api_key: str, base_url: str) -> AsyncOpenAI:
        http_client = DefaultAsyncHttpxClient(
            timeout=Timeout(
                connect=10.0,
                read=self.settings.llm_timeout,
                write=10.0,
                pool=10.0,
            ),
            limits=HttpxLimits(
                max_keepalive_connections=self.settings.llm_max_concurrency * 2,
                max_connections=self.settings.llm_max_concurrency * 3,
            ),
        )
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,  # 使用自定义智能指数退避重试
        )

    async def _call_with_retry(
        self,
        client: AsyncOpenAI,
        target_model: str,
        messages: List[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: int,
        provider_label: str,
    ) -> str:
        retries = self.settings.llm_max_retries
        start_time = time.perf_counter()

        for attempt in range(retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                latency = time.perf_counter() - start_time
                self.total_latency_seconds += latency
                self.total_success += 1
                return content.strip()

            except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
                safe_err = _sanitize_error_for_log(exc)
                is_last = attempt == retries
                if is_last:
                    logger.error(
                        f"LLM request failed on provider '{provider_label}' after {retries} retries. "
                        f"Model: {target_model}, Error: {safe_err}"
                    )
                    raise

                backoff = (2 ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    f"LLM [{provider_label}] attempt {attempt + 1} failed ({safe_err}). Retrying in {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)

            except Exception as exc:
                logger.error(
                    f"Unexpected LLM exception on provider '{provider_label}': {_sanitize_error_for_log(exc)}"
                )
                raise

        return ""

    async def chat_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """发起单次文本生成请求（带并发信号量、指数退避重试及中转站失败自动降级官方 DeepSeek）"""
        target_model = model or self.default_model
        messages: List[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with self._semaphore:
            self.total_requests += 1
            try:
                return await self._call_with_retry(
                    client=self._client,
                    target_model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider_label=self.provider_name,
                )
            except Exception as primary_exc:
                safe_primary = _sanitize_error_for_log(primary_exc)
                if self._fallback_client is not None and self._fallback_model is not None:
                    self.total_fallbacks += 1
                    logger.warning(
                        f"Relay provider '{self.provider_name}' exhausted retries ({safe_primary}); "
                        f"auto-falling back to official DeepSeek ({self._fallback_model})."
                    )
                    try:
                        return await self._call_with_retry(
                            client=self._fallback_client,
                            target_model=self._fallback_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            provider_label="deepseek(fallback)",
                        )
                    except Exception:
                        self.total_failures += 1
                        raise
                if not self.is_official and self.provider_name != "mock":
                    logger.warning(
                        f"Relay provider '{self.provider_name}' failed ({safe_primary}), "
                        "and official DeepSeek fallback is unavailable (missing valid DEEPSEEK_API_KEY)."
                    )
                self.total_failures += 1
                raise

    async def chat_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str = "You are an expert AI assistant that responds in strict JSON format.",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        model: Optional[str] = None,
    ) -> T:
        """结构化 JSON 提取并自动映射为 Pydantic 数据模型"""
        schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"【OUTPUT FORMAT REQUIREMENT】\n"
            f"Please output strictly in valid JSON matching the following JSON Schema. Do NOT include markdown comments outside the JSON object.\n"
            f"```json\n{schema_json}\n```"
        )

        raw_text = await self.chat_text(
            prompt=enhanced_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

        return self._parse_json_to_model(raw_text, response_model)

    def _parse_json_to_model(self, raw_text: str, response_model: Type[T]) -> T:
        """鲁棒解析 JSON 字符串（支持提取 ```json 块及容错修复）"""
        cleaned = raw_text.strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if json_match:
            cleaned = json_match.group(1).strip()

        try:
            return response_model.model_validate_json(cleaned)
        except Exception:
            pass

        try:
            data = json.loads(cleaned)
            return response_model.model_validate(data)
        except Exception as exc:
            brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
            if brace_match:
                try:
                    data = json.loads(brace_match.group(1))
                    return response_model.model_validate(data)
                except Exception:
                    pass
            logger.error(f"Failed to parse LLM JSON output into {response_model.__name__}. Raw text:\n{raw_text}")
            raise ValueError(f"Failed to parse LLM response into {response_model.__name__}: {exc}") from exc

    async def batch_chat(
        self,
        prompts: List[Tuple[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> List[str]:
        """批量高并发执行文本请求（受限于信号量）"""
        tasks = [
            self.chat_text(
                prompt=p,
                system_prompt=sp,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for p, sp in prompts
        ]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> Dict[str, Any]:
        """获取调用量与性能统计"""
        avg_lat = (
            round(self.total_latency_seconds / max(self.total_success, 1), 3)
            if self.total_success > 0
            else 0.0
        )
        return {
            "provider": self.provider_name,
            "model": self.default_model,
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_failures": self.total_failures,
            "total_fallbacks": self.total_fallbacks,
            "total_latency_seconds": round(self.total_latency_seconds, 2),
            "avg_latency_seconds": avg_lat,
        }

    async def close(self) -> None:
        """关闭主客户端及降级客户端的 HTTP 连接池"""
        try:
            await self._client.close()
        finally:
            if self._fallback_client is not None:
                await self._fallback_client.close()


_global_llm_client: Optional[LLMClient] = None


def get_llm_client(
    settings: Optional[Settings] = None,
    provider_name: Optional[str] = None,
) -> LLMClient:
    """
    获取 LLM 客户端：
    - 当指定了 provider_name 或非默认单例的自定义 settings 时，返回独立客户端实例（确保按请求/CLI 切换互不污染）；
    - 默认调用或传入默认 get_settings() 单例时复用全局单例，避免连接池泄漏。
    """
    global _global_llm_client
    default_settings = get_settings()
    if provider_name is not None or (settings is not None and settings is not default_settings):
        return LLMClient(settings=settings, provider_name=provider_name)
    if _global_llm_client is None:
        _global_llm_client = LLMClient(settings=default_settings)
    return _global_llm_client
