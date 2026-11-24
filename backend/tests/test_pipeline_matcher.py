# -*- coding: utf-8 -*-
"""
Unit Tests for Module 3: Node Matching, Dual-Pruned Arc Verification, and Pipeline Runner.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_client import LLMClient
from src.knowledge.cache import GraphCacheManager
from src.models.discussion import Speaker, Utterance
from src.models.graph import Arc, CandidateGraph, Node, StandardGraph
from src.pipeline.arc_verifier import ArcVerificationItem, ArcVerificationPayload, CandidateArcVerifier
from src.pipeline.node_matcher import CandidateNodeMatcher, MatchedNodeItem, NodeMatchPayload, NovelNodeItem
from src.pipeline.runner import PipelineRunner


@pytest.fixture
def mock_standard_graph():
    return StandardGraph(
        corpus_id="test_case",
        title="测试案例",
        nodes=[
            Node(id="c1", name="市场竞争激烈", type="c", description="行业红海", keywords=["竞争", "市场"]),
            Node(id="c2", name="预算紧张", type="c", description="资金有限", keywords=["预算", "资金"]),
            Node(id="d1", name="开发AI新功能", type="d", description="推出智能化模块", keywords=["AI", "新功能", "开发"]),
            Node(id="d2", name="开展全员降本", type="d", description="缩减运营成本", keywords=["降本", "缩减"]),
            Node(id="u1", name="提升市场占有率", type="u", description="增加用户与份额", keywords=["占有率", "份额"]),
        ],
        arcs=[
            Arc(source="c1", target="d1", type="info", description="市场竞争驱动AI创新"),
            Arc(source="d1", target="u1", type="causal", description="AI功能拉动份额增长"),
            Arc(source="c2", target="d2", type="info", description="预算紧张推动降本"),
        ],
    )


@pytest.fixture
def mock_speakers():
    u1 = Utterance(
        index=0,
        speaker_id="speaker_2",
        speaker_name="候选人 2",
        text="我认为目前市场竞争很激烈，我们应当尽快开发AI新功能，从而有效提升市场占有率！",
        start_time_str="01:00",
        start_seconds=60.0,
    )
    spk2 = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[u1], is_host=False)

    u2 = Utterance(
        index=1,
        speaker_id="speaker_3",
        speaker_name="候选人 3",
        text="我完全赞同2号。面对当前预算紧张的现实，我们要精打细算。",
        start_time_str="01:30",
        start_seconds=90.0,
    )
    spk3 = Speaker(speaker_id="speaker_3", display_name="候选人 3", utterances=[u2], is_host=False)

    return [spk2, spk3]


@pytest.mark.asyncio
async def test_node_matcher_llm_mock(mock_standard_graph, mock_speakers):
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(
        return_value=NodeMatchPayload(
            matched_nodes=[
                MatchedNodeItem(node_id="c1", utterance_index=0, quote="市场竞争很激烈", confidence=0.95),
                MatchedNodeItem(node_id="d1", utterance_index=0, quote="尽快开发AI新功能", confidence=0.95),
                MatchedNodeItem(node_id="u1", utterance_index=0, quote="提升市场占有率", confidence=0.9),
            ],
            novel_nodes=[
                NovelNodeItem(name="成立敏捷突击队", type="d", description="快速试错机制", quote="组建小团队"),
            ],
        )
    )

    matcher = CandidateNodeMatcher(llm_client=mock_llm)
    cand_graph = await matcher.match_nodes_for_speaker(mock_speakers[0], mock_standard_graph)

    assert len(cand_graph.matched_nodes) == 3
    assert "c1" in cand_graph.matched_node_ids
    assert "d1" in cand_graph.matched_node_ids
    assert "u1" in cand_graph.matched_node_ids
    assert len(cand_graph.node_mentions) == 3
    assert len(cand_graph.novel_nodes) == 1
    assert cand_graph.novel_nodes[0].name == "成立敏捷突击队"


@pytest.mark.asyncio
async def test_node_matcher_heuristic_fallback(mock_standard_graph, mock_speakers):
    # 模拟 LLM 抛出异常，触发降级匹配
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(side_effect=RuntimeError("LLM API Offline"))

    matcher = CandidateNodeMatcher(llm_client=mock_llm)
    cand_graph = await matcher.match_nodes_for_speaker(mock_speakers[0], mock_standard_graph)

    # 验证启发式降级是否通过关键词命中了 c1, d1, u1
    assert "c1" in cand_graph.matched_node_ids
    assert "d1" in cand_graph.matched_node_ids
    assert "u1" in cand_graph.matched_node_ids


@pytest.mark.asyncio
async def test_dual_pruned_arc_verifier(mock_standard_graph, mock_speakers):
    cand_graph = CandidateGraph(
        speaker_id="speaker_2",
        matched_nodes=[
            mock_standard_graph.get_node("c1"),
            mock_standard_graph.get_node("d1"),
            mock_standard_graph.get_node("u1"),
        ],
        node_mentions=[],
    )

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(
        return_value=ArcVerificationPayload(
            verified_arcs=[
                ArcVerificationItem(source="c1", target="d1", is_valid=True, reasoning="因果明确", weight=1.0),
                ArcVerificationItem(source="d1", target="u1", is_valid=True, reasoning="手段目的明确", weight=1.0),
            ]
        )
    )

    verifier = CandidateArcVerifier(llm_client=mock_llm)
    verified = await verifier.verify_arcs_for_speaker(mock_speakers[0], cand_graph, mock_standard_graph)

    # 验证剪枝效果：标准图谱有 3 条边，由于候选人未提及 c2/d2，c2->d2 被自动剪枝，仅向 LLM 提交 2 条边验证
    assert len(verified) == 2
    arc_ids = {f"{a.source}->{a.target}" for a in verified}
    assert "c1->d1" in arc_ids
    assert "d1->u1" in arc_ids
    assert "c2->d2" not in arc_ids


@pytest.mark.asyncio
async def test_arc_verifier_rejects_unreturned_and_false_arcs(mock_standard_graph, mock_speakers):
    """验证边校验器严格拒绝 is_valid=False 及 LLM 未返回的候选边（修复原默认放行 Bug）"""
    cand_graph = CandidateGraph(
        speaker_id="speaker_2",
        matched_nodes=[
            mock_standard_graph.get_node("c1"),
            mock_standard_graph.get_node("d1"),
            mock_standard_graph.get_node("u1"),
        ],
        node_mentions=[],
    )
    mock_llm = MagicMock(spec=LLMClient)
    # 仅返回 c1->d1 为 True，省略 d1->u1
    mock_llm.chat_json = AsyncMock(
        return_value=ArcVerificationPayload(
            verified_arcs=[
                ArcVerificationItem(source="c1", target="d1", is_valid=True),
            ]
        )
    )
    verifier = CandidateArcVerifier(llm_client=mock_llm)
    verified = await verifier.verify_arcs_for_speaker(mock_speakers[0], cand_graph, mock_standard_graph)
    assert len(verified) == 1
    assert verified[0].source == "c1" and verified[0].target == "d1"


@pytest.mark.asyncio
async def test_node_matcher_dedups_novel_against_all_std_batches_and_aggregates_windows():
    """P0-10 回归：多批标准节点时 novel_nodes 对全量标准图谱去重；多发言窗口聚合行为指标"""
    from configs.settings import Settings

    std_nodes = [
        Node(id=f"c{i}", name=f"标准概念节点_{i}", type="c")
        for i in range(1, 26)
    ]
    std_graph = StandardGraph(corpus_id="multi_batch", title="多批案例", nodes=std_nodes, arcs=[])
    # 构造 2 个发言窗口 (>7500 字)
    long_u1 = Utterance(index=0, speaker_id="speaker_2", speaker_name="候选人 2", text="讨论前期内容。" * 1200, start_seconds=10.0)
    long_u2 = Utterance(index=1, speaker_id="speaker_2", speaker_name="候选人 2", text="讨论后期提醒时间只剩五分钟。" * 1200, start_seconds=300.0)
    spk = Speaker(speaker_id="speaker_2", display_name="候选人 2", utterances=[long_u1, long_u2], is_host=False)

    mock_llm = MagicMock(spec=LLMClient)
    # 2 windows × 2 batches (batch_size=20) = 4 calls
    # Window 0 Batch 0 returns a false novel node that actually exists in Batch 1 ("标准概念节点_25")
    w0_b0 = NodeMatchPayload(
        matched_nodes=[MatchedNodeItem(node_id="c1", utterance_index=0, quote="前期")],
        novel_nodes=[
            NovelNodeItem(name="标准概念节点_25", type="c", description="其实在第二批里"),
            NovelNodeItem(name="真正独创的生态合作机制", type="d", description="真实创新"),
        ],
        openness_counts={"listening": 1, "supporting": 1, "analysis": 0, "pushing": 0},
        timing_control_count=1,
        feasibility_scores={"resource_awareness": 6.0, "risk_sensitivity": 6.0, "execution_feasibility": 6.0, "adaptability": 6.0},
    )
    w0_b1 = NodeMatchPayload(matched_nodes=[MatchedNodeItem(node_id="c25", utterance_index=0, quote="前期")])
    w1_b0 = NodeMatchPayload(
        matched_nodes=[],
        novel_nodes=[],
        openness_counts={"listening": 0, "supporting": 2, "analysis": 1, "pushing": 1},
        timing_control_count=2,
        feasibility_scores={"resource_awareness": 8.0, "risk_sensitivity": 8.0, "execution_feasibility": 8.0, "adaptability": 8.0},
    )
    w1_b1 = NodeMatchPayload(matched_nodes=[])
    mock_llm.chat_json = AsyncMock(side_effect=[w0_b0, w0_b1, w1_b0, w1_b1])

    matcher = CandidateNodeMatcher(llm_client=mock_llm, settings=Settings(node_match_batch_size=20))
    cg = await matcher.match_nodes_for_speaker(spk, std_graph)

    # 虚假创新点 "标准概念节点_25" 必须被去重剔除，仅保留真正创新点
    assert [n.name for n in cg.novel_nodes] == ["真正独创的生态合作机制"]
    # 多窗口行为指标应聚合而非只取窗口 0
    assert cg.timing_control_count == 3
    assert cg.openness_counts["supporting"] == 3
    assert cg.openness_counts["pushing"] == 1
    assert cg.feasibility_ratings["resource_awareness"] == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_arc_verifier_preserves_parallel_edge_types(mock_speakers):
    """P0-13 回归：同两点间不同 type 的平行边（如 info 与 causal）互不覆盖"""
    std_graph = StandardGraph(
        corpus_id="par",
        title="平行边测试",
        nodes=[
            Node(id="c1", name="成本上升", type="c"),
            Node(id="d1", name="优化供应链", type="d"),
        ],
        arcs=[
            Arc(source="c1", target="d1", type="info"),
            Arc(source="c1", target="d1", type="causal"),
        ],
    )
    cg = CandidateGraph(
        speaker_id="speaker_2",
        matched_nodes=list(std_graph.nodes),
    )
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(
        return_value=ArcVerificationPayload(
            verified_arcs=[
                ArcVerificationItem(source="c1", target="d1", type="info", is_valid=True),
                ArcVerificationItem(source="c1", target="d1", type="causal", is_valid=False),
            ]
        )
    )
    verifier = CandidateArcVerifier(llm_client=mock_llm)
    verified = await verifier.verify_arcs_for_speaker(mock_speakers[0], cg, std_graph)
    assert len(verified) == 1
    assert verified[0].type == "info"




@pytest.mark.asyncio
async def test_pipeline_runner_end_to_end(mock_standard_graph, tmp_path):
    # Mock LLM and Cache
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock()

    # 模拟抽取标准图谱、互动提取、节点匹配、边校验
    mock_cache = MagicMock(spec=GraphCacheManager)
    mock_cache.get_or_extract = AsyncMock(return_value=mock_standard_graph)

    runner = PipelineRunner(llm_client=mock_llm, cache_manager=mock_cache)

    sample_transcript = """
    说话人 1 00:00
    大家欢迎参加讨论。
    说话人 2 00:15
    我认为当前市场竞争激烈，必须开发AI新功能以提升市场占有率。
    说话人 3 00:45
    我赞同2号的观点，不过预算很紧张，我们需要降本增效。
    """

    result = await runner.run(
        corpus_text="测试语料内容",
        discussion_text=sample_transcript,
        group_id="test_group_1",
    )

    assert result.group_id == "test_group_1"
    assert len(result.transcript.candidate_speakers) == 2
    assert "speaker_2" in result.candidate_graphs
    assert "speaker_3" in result.candidate_graphs
    assert result.elapsed_seconds >= 0.0
