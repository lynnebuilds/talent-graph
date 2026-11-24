# -*- coding: utf-8 -*-
"""
TalentGraph Global Settings Center
Based on Pydantic V2 and pydantic-settings.
Supports Official DeepSeek API (default deepseek-flash) and arbitrary named OpenAI-compatible
relay stations via `.env` groups: RELAY_<NAME>_BASE_URL / RELAY_<NAME>_API_KEY / RELAY_<NAME>_MODEL.
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 默认顶层数据根目录: talent-graph/data
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

_RELAY_VAR_RE = re.compile(r"^RELAY_([A-Z0-9_]+)_(BASE_URL|API_KEY|MODEL)$", re.IGNORECASE)


class LLMProviderProfile(BaseModel):
    """单个 LLM 渠道的标准化连接配置（仅在服务端内部消费，严禁序列化暴露 API Key）"""

    name: str = Field(..., description="渠道名称: deepseek | relay_<name> | mock")
    base_url: str = Field(..., repr=False, description="OpenAI 兼容 API 基础地址")
    api_key: str = Field(default="", repr=False, description="渠道 API 密钥")
    model: str = Field(..., description="默认调用模型名")
    is_official: bool = Field(default=False, description="是否为官方 DeepSeek 渠道")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # 1. LLM 核心配置 (官方 DeepSeek + 任意数量 RELAY_<NAME>_* 中转站)
    # -------------------------------------------------------------------------
    llm_provider: str = Field(
        default="deepseek",
        description="激活的 LLM 渠道名称: deepseek | relay_<name> | mock",
    )

    # 官方 DeepSeek API 配置
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="官方 DeepSeek API 密钥",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="官方 DeepSeek API 基础地址",
    )
    deepseek_model: str = Field(
        default="deepseek-flash",
        validation_alias=AliasChoices("deepseek_model", "deepseek_model_name"),
        description="官方 DeepSeek 默认模型名称 (默认 deepseek-flash，可按需覆盖)",
    )

    # 通用 LLM 调用治理参数
    llm_max_concurrency: int = Field(
        default=20,
        description="LLM 最大异步并发请求数",
        ge=1,
        le=100,
    )
    llm_timeout: float = Field(
        default=60.0,
        description="LLM 请求超时时间 (秒)",
        gt=0,
    )
    llm_max_retries: int = Field(
        default=3,
        description="LLM 失败指数退避重试次数",
        ge=0,
    )

    # -------------------------------------------------------------------------
    # 2. 向量计算配置 (默认纯 CPU 本地 ONNX；可选通用 OpenAI 兼容 Embedding API)
    # -------------------------------------------------------------------------
    embedding_provider: Literal["onnx", "api", "bge", "local", "auto", "mock"] = Field(
        default="onnx",
        description="向量计算引擎提供商: onnx(本地CPU ONNX) | api(通用 OpenAI 兼容向量 API) | auto(自动优选)",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        validation_alias=AliasChoices("embedding_model", "bge_model_name"),
        description="本地 ONNX 向量模型名称或目录/文件路径",
    )
    embedding_api_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("embedding_api_base_url", "openai_base_url"),
        description="通用 OpenAI 兼容 Embedding API 基础地址（可接中转站或 OpenAI）",
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("embedding_api_key", "openai_api_key"),
        description="通用 OpenAI 兼容 Embedding API 密钥",
    )
    embedding_api_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("embedding_api_model", "openai_embedding_model"),
        description="通用 OpenAI 兼容 Embedding 模型名称",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="向量计算批次大小",
        ge=1,
    )

    # -------------------------------------------------------------------------
    # 3. 业务数据与项目目录配置 (顶层解耦与项目隔离)
    # -------------------------------------------------------------------------
    node_similarity_threshold: float = Field(
        default=0.85,
        description="节点匹配语义相似度阈值",
        ge=0.0,
        le=1.0,
    )
    corpus_auto_chunk_threshold: int = Field(
        default=4500,
        description="触发自适应分块抽取的参考语料长度阈值 (字符数)",
    )
    corpus_chunk_size: int = Field(
        default=2800,
        description="超长参考语料分块大小 (字符数)",
    )
    corpus_chunk_overlap: int = Field(
        default=300,
        description="超长参考语料分块重叠大小 (字符数)",
    )
    enable_corpus_smoothing: bool = Field(
        default=False,
        description="是否对参考语料启用高并发 ASR 语音纠错润滑 (纯文字题本默认关闭以0秒直通)",
    )
    node_match_batch_size: int = Field(
        default=70,
        description="超大规模标准图谱下候选人节点匹配的单批节点上限 (防止长Prompt漏召回或截断)",
        ge=20,
        le=200,
    )
    host_speaker_id: str = Field(
        default="auto",
        description="主持人标识: 'auto'=智能检测主持人避免误删1号选手, ''=全员参评无主持人, 或指定如 'speaker_1'",
    )
    data_root_dir: Path = Field(
        default=DEFAULT_DATA_ROOT,
        description="业务数据根目录 (与后端代码解耦，支持外部挂载)",
    )
    corpus_cache_dir: Optional[Path] = Field(
        default=None,
        description="标准题库图谱缓存路径 (默认: data/cache/corpus)",
    )
    projects_dir: Optional[Path] = Field(
        default=None,
        description="按项目/会议场次隔离归档路径 (默认: data/projects)",
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="兼容旧接口的输出路径",
    )

    # -------------------------------------------------------------------------
    # 4. Web 服务配置
    # -------------------------------------------------------------------------
    app_host: str = Field(
        default="0.0.0.0",
        description="FastAPI 监听地址",
    )
    app_port: int = Field(
        default=8000,
        description="FastAPI 监听端口",
    )
    log_level: str = Field(
        default="INFO",
        description="日志级别",
    )
    cors_origins: Union[str, List[str]] = Field(
        default="*",
        description="CORS 跨域白名单",
    )

    # -------------------------------------------------------------------------
    # 5. Playwright PDF 导出配置
    # -------------------------------------------------------------------------
    pdf_export_enabled: bool = Field(
        default=True,
        description="是否启用 Playwright PDF 导出",
    )
    pdf_timeout_ms: int = Field(
        default=30000,
        description="PDF 渲染超时时间 (毫秒)",
    )
    pdf_wait_until: Literal["load", "domcontentloaded", "networkidle", "commit"] = Field(
        default="networkidle",
        description="PDF 页面加载等待策略",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, v: object) -> str:
        text = str(v or "deepseek").strip().lower()
        return text or "deepseek"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [orig.strip() for orig in v.split(",") if orig.strip()]
        return v

    # -------------------------------------------------------------------------
    # 多渠道 LLM 发现与解析助手
    # -------------------------------------------------------------------------
    @staticmethod
    def is_valid_api_key(key: Optional[str]) -> bool:
        """判断 API Key 是否为非占位符的有效密钥"""
        if not key:
            return False
        cleaned = key.strip()
        lower = cleaned.lower()
        if not cleaned or lower in ("mock", "mock_key", "none", "null"):
            return False
        if "your_" in lower or "your-" in lower or cleaned.startswith("<"):
            return False
        return True

    def _discover_relay_raw(self) -> Dict[str, Dict[str, str]]:
        """从 .env (model_extra) 与 os.environ 自动发现所有 RELAY_<NAME>_(BASE_URL|API_KEY|MODEL) 分组"""
        merged: Dict[str, str] = {}
        # 1. 先加载 .env / 初始化传入的额外键
        if self.model_extra:
            for k, val in self.model_extra.items():
                if val is not None and _RELAY_VAR_RE.match(k):
                    merged[k.upper()] = str(val).strip()
        # 2. os.environ 覆盖 .env（与 pydantic-settings 标准优先级一致）
        for k, val in os.environ.items():
            if _RELAY_VAR_RE.match(k):
                merged[k.upper()] = val.strip()

        grouped: Dict[str, Dict[str, str]] = {}
        for key, val in merged.items():
            m = _RELAY_VAR_RE.match(key)
            if not m:
                continue
            raw_name, field_name = m.group(1).lower(), m.group(2).lower()
            relay_id = raw_name if raw_name.startswith("relay_") else f"relay_{raw_name}"
            grouped.setdefault(relay_id, {})[field_name] = val
        return grouped

    def get_official_deepseek_profile(self) -> LLMProviderProfile:
        """构造官方 DeepSeek API 渠道配置"""
        api_key = (self.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        return LLMProviderProfile(
            name="deepseek",
            base_url=self.deepseek_base_url.strip().rstrip("/"),
            api_key=api_key,
            model=self.deepseek_model.strip() or "deepseek-flash",
            is_official=True,
        )

    def get_relay_profiles(self) -> Dict[str, LLMProviderProfile]:
        """返回所有配置完整、可用的中转站渠道映射"""
        raw = self._discover_relay_raw()
        profiles: Dict[str, LLMProviderProfile] = {}
        for relay_id, fields in raw.items():
            base_url = fields.get("base_url", "").strip().rstrip("/")
            api_key = fields.get("api_key", "").strip()
            model = fields.get("model", "").strip()
            if base_url and self.is_valid_api_key(api_key) and model:
                profiles[relay_id] = LLMProviderProfile(
                    name=relay_id,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    is_official=False,
                )
        return profiles

    def get_available_llm_providers(self) -> List[str]:
        """返回当前环境已配置可用的渠道名称列表（脱敏，绝不暴露 Key/URL）"""
        available: List[str] = []
        official = self.get_official_deepseek_profile()
        if self.is_valid_api_key(official.api_key):
            available.append("deepseek")
        available.extend(sorted(self.get_relay_profiles().keys()))
        if self.llm_provider == "mock" and "mock" not in available:
            available.append("mock")
        return available

    def resolve_llm_provider(
        self,
        provider_name: Optional[str] = None,
    ) -> Tuple[LLMProviderProfile, Optional[str]]:
        """
        解析目标 LLM 渠道配置。
        若指定的中转站未配置或缺少字段，返回 (官方 DeepSeek 配置, 降级原因说明)，由调用方记录 WARNING 并自动降级。
        """
        cleaned_override = (provider_name or "").strip().lower()
        target = cleaned_override or (self.llm_provider or "deepseek").strip().lower()
        official = self.get_official_deepseek_profile()

        if target == "mock":
            return (
                LLMProviderProfile(
                    name="mock",
                    base_url=official.base_url,
                    api_key="mock_key",
                    model=official.model,
                    is_official=False,
                ),
                None,
            )

        if target == "deepseek":
            return official, None

        relays = self.get_relay_profiles()
        norm_id = target if target.startswith("relay_") else f"relay_{target}"
        if norm_id in relays:
            return relays[norm_id], None

        raw_relays = self._discover_relay_raw()
        if norm_id in raw_relays:
            fields = raw_relays[norm_id]
            missing = []
            if not fields.get("base_url", "").strip():
                missing.append("BASE_URL")
            if not self.is_valid_api_key(fields.get("api_key")):
                missing.append("API_KEY")
            if not fields.get("model", "").strip():
                missing.append("MODEL")
            reason = (
                f"Relay provider '{norm_id}' is incomplete (missing {', '.join(missing)}); "
                "falling back to official 'deepseek'."
            )
        else:
            reason = f"Relay provider '{target}' is not configured; falling back to official 'deepseek'."
        return official, reason

    # -------------------------------------------------------------------------
    # 运行时路径助手
    # -------------------------------------------------------------------------
    def get_corpus_cache_dir(self) -> Path:
        return self.corpus_cache_dir or (self.data_root_dir / "cache" / "corpus")

    def get_projects_dir(self) -> Path:
        return self.projects_dir or self.output_dir or (self.data_root_dir / "projects")

    def get_project_dir(self, project_id: str) -> Path:
        """获取指定项目/会议场次的独立归档目录: data/projects/{project_id}"""
        return self.get_projects_dir() / project_id

    def get_project_report_file(self, project_id: str) -> Path:
        """获取项目全量报告数据文件: data/projects/{project_id}/report.json"""
        return self.get_project_dir(project_id) / "report.json"

    def get_project_graphs_dir(self, project_id: str) -> Path:
        """获取项目拓扑图谱子目录: data/projects/{project_id}/graphs"""
        return self.get_project_dir(project_id) / "graphs"

    def get_project_pdf_dir(self, project_id: str) -> Path:
        """获取项目 PDF 报告导出子目录: data/projects/{project_id}/pdf_reports"""
        return self.get_project_dir(project_id) / "pdf_reports"

    def ensure_directories(self) -> None:
        """确保运行时核心目录结构存在"""
        self.get_corpus_cache_dir().mkdir(parents=True, exist_ok=True)
        self.get_projects_dir().mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """获取全局单例配置实例（带 LRU 缓存）"""
    s = Settings()
    s.ensure_directories()
    return s
