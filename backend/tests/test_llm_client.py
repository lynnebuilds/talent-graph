# -*- coding: utf-8 -*-
"""Unit tests for Multi-Provider LLM Client & Official DeepSeek Fallback"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError
from pydantic import BaseModel, Field

try:
    import httpx2 as httpx_mod
except ImportError:  # pragma: no cover - fallback for openai 1.x environments
    import httpx as httpx_mod  # type: ignore[no-redef]

from configs.settings import Settings, get_settings
from src.core.llm_client import LLMClient, _sanitize_error_for_log, get_llm_client


class MockSampleResponse(BaseModel):
    summary: str
    score: int = Field(default=100)


@pytest.mark.asyncio
async def test_json_parsing_robustness():
    """测试各种格式的 JSON 解析容错能力"""
    client = LLMClient(Settings(_env_file=None, deepseek_api_key="mock_key"))

    # 1. 正常标准 JSON
    res1 = client._parse_json_to_model('{"summary": "test1", "score": 90}', MockSampleResponse)
    assert res1.summary == "test1"
    assert res1.score == 90

    # 2. 带 markdown ```json 代码块
    res2 = client._parse_json_to_model('```json\n{"summary": "test2", "score": 85}\n```', MockSampleResponse)
    assert res2.summary == "test2"
    assert res2.score == 85

    # 3. 前后带自然语言噪声文本
    res3 = client._parse_json_to_model('Here is the result:\n```json\n{"summary": "test3", "score": 95}\n```\nHope this helps!', MockSampleResponse)
    assert res3.summary == "test3"
    assert res3.score == 95

    # 4. 纯花括号内嵌在文本中
    res4 = client._parse_json_to_model('Output is {"summary": "test4", "score": 70} completed.', MockSampleResponse)
    assert res4.summary == "test4"
    assert res4.score == 70


@pytest.mark.asyncio
async def test_llm_client_mock_chat():
    """测试通过 Mock OpenAI 接口验证 chat_text 与 chat_json 流程"""
    client = LLMClient(Settings(_env_file=None, deepseek_api_key="mock_key", llm_max_concurrency=5))

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"summary": "Great analysis", "score": 98}\n```'
    mock_completion.choices = [mock_choice]

    with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)):
        # 测试 chat_text
        text = await client.chat_text("Hello")
        assert "Great analysis" in text

        # 测试 chat_json
        parsed = await client.chat_json("Hello", MockSampleResponse)
        assert parsed.summary == "Great analysis"
        assert parsed.score == 98

    stats = client.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_success"] == 2


def test_llm_client_selects_official_deepseek_and_multiple_relays(monkeypatch):
    """支持官方 DeepSeek (deepseek-flash) 与多个中转站按需切换，互不污染"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-official-key")
    monkeypatch.setenv("RELAY_A_BASE_URL", "https://relay-a.test/v1")
    monkeypatch.setenv("RELAY_A_API_KEY", "sk-relay-a")
    monkeypatch.setenv("RELAY_A_MODEL", "deepseek-v3-relay")
    monkeypatch.setenv("RELAY_B_BASE_URL", "https://relay-b.test/v1")
    monkeypatch.setenv("RELAY_B_API_KEY", "sk-relay-b")
    monkeypatch.setenv("RELAY_B_MODEL", "claude-proxy")

    settings = Settings(_env_file=None, llm_provider="deepseek")

    c_official = get_llm_client(settings, provider_name="deepseek")
    c_relay_a = get_llm_client(settings, provider_name="relay_a")
    c_relay_b = get_llm_client(settings, provider_name="relay_b")

    assert c_official.provider_name == "deepseek"
    assert c_official.default_model == "deepseek-flash"
    assert c_relay_a.provider_name == "relay_a"
    assert c_relay_a.default_model == "deepseek-v3-relay"
    assert c_relay_b.provider_name == "relay_b"
    assert c_relay_b.default_model == "claude-proxy"

    # 缺失的中转站初始化时自动降级到官方 DeepSeek
    c_missing = LLMClient(settings, provider_name="relay_nonexistent")
    assert c_missing.provider_name == "deepseek"
    assert c_missing.default_model == "deepseek-flash"
    assert c_missing.fallback_reason is not None


@pytest.mark.asyncio
async def test_relay_runtime_failure_auto_falls_back_to_official_deepseek(monkeypatch):
    """中转站运行时重试耗尽后，自动降级切换官方 DeepSeek 客户端重试并成功返回"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-official-real-key")
    monkeypatch.setenv("RELAY_A_BASE_URL", "https://relay-a.test/v1")
    monkeypatch.setenv("RELAY_A_API_KEY", "sk-relay-a-key")
    monkeypatch.setenv("RELAY_A_MODEL", "relay-model-x")

    settings = Settings(_env_file=None, llm_provider="relay_a", llm_max_retries=1)
    client = LLMClient(settings)
    assert client.provider_name == "relay_a"

    conn_err = APIConnectionError(request=httpx_mod.Request("POST", "https://relay-a.test/v1/chat/completions"))

    fallback_completion = MagicMock()
    fallback_choice = MagicMock()
    fallback_choice.message.content = "official deepseek fallback ok"
    fallback_completion.choices = [fallback_choice]

    primary_create = AsyncMock(side_effect=conn_err)
    fallback_create = AsyncMock(return_value=fallback_completion)
    assert client._fallback_client is not None

    with (
        patch.object(client._client.chat.completions, "create", new=primary_create),
        patch.object(client._fallback_client.chat.completions, "create", new=fallback_create),
        patch("src.core.llm_client.asyncio.sleep", new=AsyncMock()),
    ):
        result = await client.chat_text("测试自动降级")

    assert result == "official deepseek fallback ok"
    # llm_max_retries=1 -> 初始 1 次 + 重试 1 次 = 2 次耗尽后触发降级
    assert primary_create.await_count == 2
    fallback_create.assert_awaited_once()
    # 验证降级调用使用了官方 DeepSeek 的默认模型 deepseek-flash
    assert fallback_create.call_args.kwargs["model"] == "deepseek-flash"
    assert client.get_stats()["total_fallbacks"] == 1
    await client.close()


@pytest.mark.asyncio
async def test_official_deepseek_failure_does_not_self_recurse(monkeypatch):
    """官方 DeepSeek 本身失败时直接抛错，不触发自递归死循环"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-official-real-key")
    settings = Settings(_env_file=None, llm_provider="deepseek", llm_max_retries=0)
    client = LLMClient(settings)
    assert client._fallback_client is None

    conn_err = APIConnectionError(request=httpx_mod.Request("POST", "https://api.deepseek.com/v1/chat/completions"))
    with patch.object(client._client.chat.completions, "create", new=AsyncMock(side_effect=conn_err)):
        with pytest.raises(APIConnectionError):
            await client.chat_text("直接失败")
    await client.close()


@pytest.mark.asyncio
async def test_singleton_reuse_log_redaction_and_safe_close(monkeypatch):
    """验证默认单例复用、日志 URL/Key 脱敏及双连接池安全关闭"""
    default_settings = get_settings()
    assert get_llm_client(default_settings) is get_llm_client()

    # 验证异常脱敏不泄漏私有中转站 URL 与 API Key
    leaky_err = RuntimeError("Failed POST https://private-relay.internal/v1/chat with key sk-secret123456")
    redacted = _sanitize_error_for_log(leaky_err)
    assert "private-relay.internal" not in redacted
    assert "sk-secret123456" not in redacted
    assert "[REDACTED_URL]" in redacted
    assert "[REDACTED_KEY]" in redacted

    # 验证即使主客户端 close() 异常，降级客户端仍会被关闭
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-official-close-test")
    monkeypatch.setenv("RELAY_C_BASE_URL", "https://relay-c.test/v1")
    monkeypatch.setenv("RELAY_C_API_KEY", "sk-relay-c-test")
    monkeypatch.setenv("RELAY_C_MODEL", "deepseek-chat")
    client = LLMClient(Settings(_env_file=None, llm_provider="relay_c"))
    assert client._fallback_client is not None

    fb_close = AsyncMock()
    with (
        patch.object(client._client, "close", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch.object(client._fallback_client, "close", new=fb_close),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await client.close()
    fb_close.assert_awaited_once()
