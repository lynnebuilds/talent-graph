# -*- coding: utf-8 -*-
"""Unit tests for configuration center (Official DeepSeek + multi-relay discovery + legacy cleanup)"""

from pathlib import Path

from configs.settings import Settings


def test_default_settings_and_legacy_cleanup(monkeypatch):
    """原生默认配置：DeepSeek 默认模型为 deepseek-flash，且已彻底移除前司/历史冗余字段"""
    for env_key in ("LLM_PROVIDER", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_MODEL_NAME"):
        monkeypatch.delenv(env_key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "deepseek"
    assert settings.deepseek_base_url == "https://api.deepseek.com/v1"
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.llm_max_concurrency == 20
    assert isinstance(settings.cors_origins, list)

    # 确认前司/历史冗余配置已彻底移除
    assert not hasattr(settings, "dashscope_api_key")
    assert not hasattr(settings, "qwen_model_name")
    assert not hasattr(settings, "dashscope_embedding_model")
    assert not hasattr(settings, "mock_email")
    assert not hasattr(settings, "enable_telemetry")

    # 通用 OpenAI 兼容 Embedding 配置存在
    assert settings.embedding_provider == "onnx"
    assert settings.embedding_api_base_url == "https://api.openai.com/v1"
    assert settings.embedding_api_model == "text-embedding-3-small"


def test_multi_relay_discovery_from_env_and_fallback(monkeypatch):
    """支持 .env / 环境变量定义任意多中转站，完整中转站可切换，缺失或不完整自动降级官方 DeepSeek"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-official-deepseek-real")
    monkeypatch.setenv("RELAY_A_BASE_URL", "https://relay-a.example.com/v1")
    monkeypatch.setenv("RELAY_A_API_KEY", "sk-relay-a-secret")
    monkeypatch.setenv("RELAY_A_MODEL", "deepseek-chat-relay-a")
    monkeypatch.setenv("RELAY_B_BASE_URL", "https://relay-b.example.com/v1")
    monkeypatch.setenv("RELAY_B_API_KEY", "sk-relay-b-secret")
    monkeypatch.setenv("RELAY_B_MODEL", "gpt-4o-mini")
    # 不完整中转站（缺 API_KEY）
    monkeypatch.setenv("RELAY_BROKEN_BASE_URL", "https://broken.example.com/v1")
    monkeypatch.setenv("RELAY_BROKEN_MODEL", "some-model")

    settings = Settings(_env_file=None, llm_provider="RELAY_A")
    assert settings.llm_provider == "relay_a"

    available = settings.get_available_llm_providers()
    assert "deepseek" in available
    assert "relay_a" in available
    assert "relay_b" in available
    assert "relay_broken" not in available

    # 1. 正常命中 relay_a
    prof_a, warn_a = settings.resolve_llm_provider("relay_a")
    assert warn_a is None
    assert prof_a.name == "relay_a"
    assert prof_a.base_url == "https://relay-a.example.com/v1"
    assert prof_a.api_key == "sk-relay-a-secret"
    assert prof_a.model == "deepseek-chat-relay-a"
    assert prof_a.is_official is False

    # 2. 按需切换到 relay_b
    prof_b, warn_b = settings.resolve_llm_provider("relay_b")
    assert warn_b is None
    assert prof_b.name == "relay_b"
    assert prof_b.model == "gpt-4o-mini"

    # 3. 不完整中转站自动降级到官方 DeepSeek，并返回明确降级原因
    prof_broken, warn_broken = settings.resolve_llm_provider("relay_broken")
    assert warn_broken is not None
    assert "relay_broken" in warn_broken
    assert prof_broken.name == "deepseek"
    assert prof_broken.is_official is True
    assert prof_broken.model == "deepseek-flash"

    # 4. 未配置的中转站自动降级到官方 DeepSeek
    prof_unknown, warn_unknown = settings.resolve_llm_provider("relay_not_exist")
    assert warn_unknown is not None
    assert prof_unknown.name == "deepseek"
    assert prof_unknown.is_official is True

    # 5. 纯空白覆盖值回退到配置的默认渠道（而非触发无效空渠道降级）
    prof_blank, warn_blank = settings.resolve_llm_provider("   ")
    assert warn_blank is None
    assert prof_blank.name == "relay_a"

    # 6. Profile repr 隐藏敏感 api_key 与私有中转站 base_url
    prof_repr = repr(prof_a)
    assert "sk-relay-a-secret" not in prof_repr
    assert "relay-a.example.com" not in prof_repr


def test_multi_relay_discovery_from_dotenv_file(tmp_path: Path, monkeypatch):
    """验证直接从 .env 文件加载 RELAY_<NAME>_* 分组"""
    monkeypatch.delenv("RELAY_FAST_BASE_URL", raising=False)
    monkeypatch.delenv("RELAY_FAST_API_KEY", raising=False)
    monkeypatch.delenv("RELAY_FAST_MODEL", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=relay_fast\n"
        "DEEPSEEK_API_KEY=sk-official-123\n"
        "RELAY_FAST_BASE_URL=https://fast.relay.dev/v1\n"
        "RELAY_FAST_API_KEY=sk-fast-456\n"
        "RELAY_FAST_MODEL=deepseek-flash\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    prof, warn = settings.resolve_llm_provider()
    assert warn is None
    assert prof.name == "relay_fast"
    assert prof.base_url == "https://fast.relay.dev/v1"
    assert prof.api_key == "sk-fast-456"
    assert prof.model == "deepseek-flash"


def test_cors_origins_parsing():
    s = Settings(_env_file=None, cors_origins="http://localhost:3000, http://localhost:8000")
    assert s.cors_origins == ["http://localhost:3000", "http://localhost:8000"]

    s_star = Settings(_env_file=None, cors_origins="*")
    assert s_star.cors_origins == ["*"]


def test_directory_creation(tmp_path: Path):
    cache_dir = tmp_path / "test_cache"
    projects_dir = tmp_path / "test_projects"
    s = Settings(_env_file=None, corpus_cache_dir=cache_dir, projects_dir=projects_dir)
    s.ensure_directories()
    assert cache_dir.exists()
    assert projects_dir.exists()
    assert s.get_project_dir("p1").name == "p1"
    assert s.get_project_report_file("p1").name == "report.json"
    assert s.get_project_graphs_dir("p1").name == "graphs"
    assert s.get_project_pdf_dir("p1").name == "pdf_reports"
