# -*- coding: utf-8 -*-
"""
Unit Tests for Module 5: Diagnostic Summarizer, Report Generator, and FastAPI Endpoints.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from server import app
from src.core.llm_client import LLMClient
from src.models.discussion import DiscussionTranscript, InteractionNetwork, Speaker, Utterance
from src.models.graph import CandidateGraph, Node, StandardGraph
from src.models.report import ComprehensiveDiagnosis, ReportData
from src.pipeline.runner import PipelineResult
from src.reporting.generator import ReportGenerator
from src.reporting.summarizer import CandidateSummarizer
from src.scoring.base import ScoringContext
from src.scoring.engine import ScoringEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "talent_graph"


@pytest.mark.asyncio
async def test_candidate_summarizer_heuristic_and_mock():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(
        return_value=ComprehensiveDiagnosis(
            summary="候选人表现沉稳，逻辑严密",
            strengths=["系统思维全面", "主导核心议题收敛"],
            weaknesses=["缺少对细节成本的考量"],
            growth_suggestions=["多关注底层执行边界"],
            tags=["战略领航者", "严谨推演派"],
        )
    )

    summarizer = CandidateSummarizer(llm_client=mock_llm)

    u1 = Utterance(index=0, speaker_id="speaker_2", speaker_name="候选人 2", text="我的方案如下...", start_time_str="01:00", start_seconds=60.0)
    spk2 = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[u1], is_host=False)

    engine = ScoringEngine()
    std_graph = StandardGraph(corpus_id="c1", title="案例", nodes=[Node(id="c1", name="节点", type="c")], arcs=[])
    transcript = DiscussionTranscript(group_id="group_1", title="讨论", speakers={"speaker_2": spk2}, utterances=[u1])
    interaction = InteractionNetwork(speakers=["speaker_2"], edges=[])

    context = ScoringContext(
        group_id="group_1",
        corpus_id="c1",
        standard_graph=std_graph,
        transcript=transcript,
        candidate_graphs={"speaker_2": CandidateGraph(speaker_id="speaker_2", matched_nodes=[std_graph.nodes[0]])},
        interaction_network=interaction,
    )
    score_matrix = engine.evaluate_all(context)

    diag = await summarizer.generate_diagnosis(spk2, score_matrix.participants["speaker_2"], score_matrix)
    assert len(diag.strengths) == 2
    assert "战略领航者" in diag.tags
    assert len(diag.summary) > 0


@pytest.mark.asyncio
async def test_report_generator_end_to_end(tmp_path):
    u1 = Utterance(index=0, speaker_id="speaker_2", speaker_name="候选人 2", text="我们应当快速行动。", start_time_str="01:00", start_seconds=60.0)
    spk2 = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[u1], is_host=False)
    std_node = Node(id="c1", name="背景", type="c")
    std_graph = StandardGraph(corpus_id="c1", title="案例大纲", nodes=[std_node], arcs=[])
    transcript = DiscussionTranscript(group_id="test_g", title="测试讨论", speakers={"speaker_2": spk2}, utterances=[u1])
    interaction = InteractionNetwork(speakers=["speaker_2"], edges=[])

    pipeline_res = PipelineResult(
        group_id="test_g",
        corpus_id="c1",
        standard_graph=std_graph,
        transcript=transcript,
        candidate_graphs={"speaker_2": CandidateGraph(speaker_id="speaker_2", matched_nodes=[std_node])},
        interaction_network=interaction,
        elapsed_seconds=1.2,
    )

    mock_summarizer = MagicMock(spec=CandidateSummarizer)
    mock_summarizer.summarize_all_participants = AsyncMock(
        return_value={"speaker_2": ComprehensiveDiagnosis(summary="整体优秀", strengths=["表现优秀"], tags=["领航者"])}
    )

    generator = ReportGenerator(summarizer=mock_summarizer)
    report_data: ReportData = await generator.generate_report(pipeline_res, task_id="task_unit_test")

    assert report_data.task_id == "task_unit_test"
    assert report_data.team_overview.participant_count == 1
    assert len(report_data.participants) == 1
    assert report_data.participants[0].speaker_id == "speaker_2"
    assert len(report_data.participants[0].radar_data) == 16


def test_assessment_endpoints(client):
    """测试评估提交与数据获取端点"""
    submit_resp = client.post(
        "/api/v1/assessment/submit",
        json={
            "corpus_text": "这是一个测试案例参考语料",
            "discussion_text": "说话人 1 00:01 我们开始讨论方案。\n说话人 2 00:15 我同意这个方案。",
            "group_id": "test_g1",
            "title": "测试评估"
        }
    )
    assert submit_resp.status_code == 200
    task_id = submit_resp.json()["task_id"]
    assert task_id.startswith("task_")

    # 查询任务状态
    status_resp = client.get(f"/api/v1/assessment/{task_id}")
    assert status_resp.status_code == 200

    # 检查打印模板路由
    print_resp = client.get(f"/report-print/{task_id}/speaker_2")
    assert print_resp.status_code == 200


def test_health_exposes_sanitized_providers_and_submit_supports_per_request_provider(client, monkeypatch):
    """验证 /health 返回脱敏可用渠道清单，且 /submit 支持按次 llm_provider 覆盖不泄漏密钥"""
    from unittest.mock import patch

    from configs.settings import Settings
    from src.api.v1.assessment import AssessmentSubmitRequest

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-official-999")
    monkeypatch.setenv("RELAY_ONE_BASE_URL", "https://one.relay.test/v1")
    monkeypatch.setenv("RELAY_ONE_API_KEY", "sk-secret-relay-888")
    monkeypatch.setenv("RELAY_ONE_MODEL", "deepseek-chat")

    fresh_settings = Settings(_env_file=None, llm_provider="deepseek")
    with patch("server.get_settings", return_value=fresh_settings):
        resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["llm_provider"] == "deepseek"
    assert payload["llm_model"] == "deepseek-flash"
    assert "deepseek" in payload["available_llm_providers"]
    assert "relay_one" in payload["available_llm_providers"]
    # 严禁暴露任何 API Key 或 URL
    raw_json = resp.text
    assert "sk-secret-official-999" not in raw_json
    assert "sk-secret-relay-888" not in raw_json
    assert "one.relay.test" not in raw_json

    # 验证请求契约支持可选 llm_provider 覆盖
    req = AssessmentSubmitRequest(
        corpus_text="语料",
        discussion_text="说话人 1 00:01 发言",
        llm_provider="relay_one",
    )
    assert req.llm_provider == "relay_one"


@pytest.mark.asyncio
async def test_run_assessment_task_uses_isolated_provider_without_mutating_global(monkeypatch):
    """验证 run_assessment_task 按次构造独立 LLMClient 覆盖渠道且不污染全局配置"""
    from unittest.mock import patch

    from configs.settings import Settings
    from src.api.v1.assessment import AssessmentSubmitRequest, run_assessment_task

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-official-iso-123")
    monkeypatch.setenv("RELAY_ONE_BASE_URL", "https://iso.relay.test/v1")
    monkeypatch.setenv("RELAY_ONE_API_KEY", "sk-relay-iso-456")
    monkeypatch.setenv("RELAY_ONE_MODEL", "deepseek-chat")

    base_settings = Settings(_env_file=None, llm_provider="deepseek")
    captured = {}

    class DummyRunner:
        def __init__(self, settings=None, llm_client=None, **_kwargs):
            captured["runner_provider"] = llm_client.provider_name
            captured["runner_model"] = llm_client.default_model

        async def run(self, **_kwargs):
            return MagicMock(group_id="g_iso")

    class DummyReportGen:
        def __init__(self, settings=None, llm_client=None, **_kwargs):
            captured["report_provider"] = llm_client.provider_name
            captured["report_client"] = llm_client

        async def generate_report(self, **_kwargs):
            return MagicMock(group_id="g_iso")

    req = AssessmentSubmitRequest(
        corpus_text="测试语料",
        discussion_text="说话人 1 00:01 开始讨论",
        llm_provider="relay_one",
    )

    with (
        patch("src.api.v1.assessment.get_settings", return_value=base_settings),
        patch("src.api.v1.assessment.PipelineRunner", DummyRunner),
        patch("src.api.v1.assessment.ReportGenerator", DummyReportGen),
        patch.object(LLMClient, "close", new_callable=AsyncMock) as mock_close,
    ):
        await run_assessment_task("task_iso_test", req)

    assert captured["runner_provider"] == "relay_one"
    assert captured["runner_model"] == "deepseek-chat"
    assert captured["report_provider"] == "relay_one"
    mock_close.assert_awaited_once()
    # 全局设置保持 deepseek 未被篡改
    assert base_settings.llm_provider == "deepseek"

    # 验证即使 PipelineRunner 初始化抛异常，也会安全关闭连接池并将任务标记为 failed
    from src.api.v1.assessment import TASK_STORE

    class FailingRunner:
        def __init__(self, **_kwargs):
            raise RuntimeError("init boom")

    with (
        patch("src.api.v1.assessment.get_settings", return_value=base_settings),
        patch("src.api.v1.assessment.PipelineRunner", FailingRunner),
        patch.object(LLMClient, "close", new_callable=AsyncMock) as mock_close_fail,
    ):
        await run_assessment_task("task_iso_fail", req)

    mock_close_fail.assert_awaited_once()
    assert TASK_STORE["task_iso_fail"]["status"] == "failed"



