# -*- coding: utf-8 -*-
"""
V5 Mathematical & Behavioral Parity Regression Suite.
Locks in core invariants audited against legacy v5_speed so v5_speed can be safely retired.
"""

from unittest.mock import MagicMock

import pytest

from src.models.discussion import DiscussionTranscript, InteractionEdge, InteractionNetwork, Speaker, Utterance
from src.models.graph import Arc, CandidateGraph, Node, NodeMention, StandardGraph
from src.models.score import ALL_DIMENSIONS, DimensionScore, ParticipantScoreProfile, ScoreMatrix
from src.scoring.base import ScoringContext
from src.scoring.dimensions.d1_thinking import AbstractionScorer, LogicalReasoningScorer, SystematicThinkingScorer
from src.scoring.dimensions.d2_execution import GoalDecompositionScorer, ImpactPredictionScorer, InfoCollectionScorer
from src.scoring.dimensions.d3_creativity import AssociationScorer, CreativityScorer, FeasibilityScorer
from src.scoring.dimensions.d4_influence import InfluenceScorer
from src.scoring.dimensions.d5_teamwork import CollaborationCentralityScorer, ConflictResolutionScorer, OpennessScorer
from src.scoring.dimensions.d6_efficiency import ConvergenceScorer, FocusScorer, TimingControlScorer
from src.scoring.engine import ScoringEngine


def _make_speaker(spk_id: str, texts: list[str], times: list[float] | None = None) -> Speaker:
    utts = []
    for idx, txt in enumerate(texts):
        ts = times[idx] if times else float(idx * 30)
        utts.append(
            Utterance(
                index=idx,
                speaker_id=spk_id,
                speaker_name=spk_id,
                text=txt,
                start_seconds=ts,
                end_seconds=ts + 20.0,
            )
        )
    return Speaker(
        speaker_id=spk_id,
        display_name=spk_id,
        utterances=utts,
        is_host=False,
        total_duration_seconds=20.0 * len(utts),
        turn_count=len(utts),
    )


def test_competition_tie_ranking_and_v5_six_category_total_weight():
    """P0-1 & P1 总分：同分必须并列排名 (1, 1, 3)；总分采用 V5 六大类等权均值（维度 4 占 1/6）"""
    matrix = ScoreMatrix(group_id="tie_test")

    def _profile(spk: str, score_dim4: float, score_other: float) -> ParticipantScoreProfile:
        dims = {}
        for d in ALL_DIMENSIONS:
            sc = score_dim4 if d.id == "4" else score_other
            dims[d.id] = DimensionScore(dimension_id=d.id, dimension_name=d.name, score=sc)
        return ParticipantScoreProfile(speaker_id=spk, display_name=spk, dimension_scores=dims)

    # spk1 与 spk2 完全同分，应并列第 1，spk3 为第 3
    matrix.participants["spk1"] = _profile("spk1", 90.0, 60.0)
    matrix.participants["spk2"] = _profile("spk2", 90.0, 60.0)
    matrix.participants["spk3"] = _profile("spk3", 60.0, 60.0)

    engine = ScoringEngine()
    for p in matrix.participants.values():
        p.total_score = engine.compute_total_score(p.dimension_scores)
    matrix.compute_aggregations()

    # 六大类等权均值: (60 + 60 + 60 + 90 + 60 + 60) / 6 = 65.0 (而非 16 维平权的 61.9)
    assert matrix.participants["spk1"].total_score == pytest.approx(65.0, abs=0.1)
    assert matrix.participants["spk1"].overall_rank == 1
    assert matrix.participants["spk2"].overall_rank == 1
    assert matrix.participants["spk3"].overall_rank == 3
    assert matrix.participants["spk1"].dimension_scores["4"].rank == 1
    assert matrix.participants["spk2"].dimension_scores["4"].rank == 1
    assert matrix.participants["spk3"].dimension_scores["4"].rank == 3


def test_1_1_deduction_ff_gate_and_monotonic_verified_arcs():
    """P0-6：1_1 纯 f->f 无前置概念推导不计演绎；从 0 条验证边增至 1 条不应导致分数骤降"""
    f1 = Node(id="c1", name="事实1", type="c", level="三级")
    f2 = Node(id="c2", name="事实2", type="c", level="三级")
    c_top = Node(id="c3", name="上位概念", type="c", level="二级")
    std = StandardGraph(
        corpus_id="g",
        nodes=[f1, f2, c_top],
        arcs=[
            Arc(source="c1", target="c2", type="causal"),
            Arc(source="c3", target="c1", type="causal"),
        ],
    )
    spk = _make_speaker("s1", ["事实1 事实2"])
    transcript = DiscussionTranscript(group_id="g", speakers={"s1": spk}, utterances=spk.utterances)

    # 仅有 f1->f2（无前置概念 c3），演绎应为 0
    cg_ff_only = CandidateGraph(
        speaker_id="s1",
        matched_nodes=[f1, f2],
        node_mentions=[
            NodeMention(node_id="c1", utterance_index=0, timestamp_seconds=0.0),
            NodeMention(node_id="c2", utterance_index=0, timestamp_seconds=0.0),
        ],
        verified_arcs=[Arc(source="c1", target="c2", type="causal")],
    )
    ctx_ff = ScoringContext(
        group_id="g", corpus_id="g", standard_graph=std, transcript=transcript,
        candidate_graphs={"s1": cg_ff_only}, interaction_network=InteractionNetwork(),
    )
    raw_ff, metrics_ff, _ = LogicalReasoningScorer().calculate_raw_metrics("s1", ctx_ff)
    assert metrics_ff["deduction_count"] == 0
    assert raw_ff == 0.0

    # 0 条验证边 <= 1 条有效验证边（单调不降）
    cg_zero = CandidateGraph(
        speaker_id="s1",
        matched_nodes=[c_top, f1, f2],
        node_mentions=[
            NodeMention(node_id="c3", utterance_index=0, timestamp_seconds=0.0),
            NodeMention(node_id="c1", utterance_index=0, timestamp_seconds=0.0),
            NodeMention(node_id="c2", utterance_index=0, timestamp_seconds=0.0),
        ],
        verified_arcs=[],
    )
    cg_one = cg_zero.model_copy(update={"verified_arcs": [Arc(source="c3", target="c1", type="causal")]})
    ctx_zero = ctx_ff.model_copy(update={"candidate_graphs": {"s1": cg_zero}})
    ctx_one = ctx_ff.model_copy(update={"candidate_graphs": {"s1": cg_one}})
    raw_zero, _, _ = LogicalReasoningScorer().calculate_raw_metrics("s1", ctx_zero)
    raw_one, _, _ = LogicalReasoningScorer().calculate_raw_metrics("s1", ctx_one)
    assert raw_one >= raw_zero
    assert raw_one >= 1.0


def test_1_2_requires_causal_graph_membership_and_1_3_zero_collapse_no_density():
    """1_2 无因果边图谱返回 0；1_3 缺二级或三级节点严格归 0 且不受小图 density 虚高影响"""
    n1 = Node(id="c1", name="子事实1", type="c", level="三级", parent_concept="概念A")
    n2 = Node(id="c2", name="子事实2", type="c", level="三级", parent_concept="概念A")
    std_no_arcs = StandardGraph(corpus_id="g", nodes=[n1, n2], arcs=[])
    spk = _make_speaker("s1", ["发言"])
    transcript = DiscussionTranscript(group_id="g", speakers={"s1": spk}, utterances=spk.utterances)
    cg = CandidateGraph(
        speaker_id="s1",
        matched_nodes=[n1, n2],
        node_mentions=[
            NodeMention(node_id="c1", utterance_index=0, timestamp_seconds=0.0),
            NodeMention(node_id="c2", utterance_index=0, timestamp_seconds=0.0),
        ],
        novel_nodes=[Node(id="nov1", name="创新点", type="d", level="三级")],
    )
    ctx = ScoringContext(
        group_id="g", corpus_id="g", standard_graph=std_no_arcs, transcript=transcript,
        candidate_graphs={"s1": cg}, interaction_network=InteractionNetwork(),
    )
    raw_12, _, _ = AbstractionScorer().calculate_raw_metrics("s1", ctx)
    assert raw_12 == 0.0

    # 1_3: 仅有三级无二级，按 V5 公式 (1+0)*0*2 = 0，不应被 density 或覆盖项抬成正分
    raw_13, _, _ = SystematicThinkingScorer().calculate_raw_metrics("s1", ctx)
    assert raw_13 == 0.0


def test_2_1_boolean_fallback_zero_goals_and_2_2_pure_memory_no_live_embed():
    """P0-5 & P0-7：2_1 有显式 final 且无 inst 时不覆盖标注、0 目标得 0；2_2 不调用实时 embed_texts"""
    u_fin = Node(id="u1", name="利润增长", type="u", sub_type="final")
    u_other = Node(id="u2", name="品牌提升", type="u", sub_type="final")
    d1 = Node(id="d1", name="推出新品", type="d")
    c1 = Node(id="c1", name="市场规模", type="c")
    c2 = Node(id="c2", name="成本压力", type="c")
    std = StandardGraph(
        corpus_id="g",
        nodes=[c1, c2, d1, u_fin, u_other],
        arcs=[
            Arc(source="u1", target="u2", type="causal"),
            Arc(source="d1", target="u1", type="causal"),
        ],
    )
    spk = _make_speaker("s1", ["推出新品"])
    transcript = DiscussionTranscript(group_id="g", speakers={"s1": spk}, utterances=spk.utterances)

    # 仅提及 d1 (0 目标) -> 2_1 应严格为 0
    cg_no_goal = CandidateGraph(speaker_id="s1", matched_nodes=[d1, c1, c2])
    mock_emb = MagicMock()
    mock_emb.embed_texts.side_effect = AssertionError("2_2 scorer must NOT call live embed_texts!")
    ctx = ScoringContext(
        group_id="g", corpus_id="g", standard_graph=std, transcript=transcript,
        candidate_graphs={"s1": cg_no_goal}, interaction_network=InteractionNetwork(),
        embedding_engine=mock_emb,
    )
    raw_21, _, _ = GoalDecompositionScorer().calculate_raw_metrics("s1", ctx)
    assert raw_21 == 0.0

    # 显式标注两个 final 时，u1 不应被拓扑推断改写为 instrumental
    cg_with_u1 = CandidateGraph(speaker_id="s1", matched_nodes=[u_fin])
    ctx_u1 = ctx.model_copy(update={"candidate_graphs": {"s1": cg_with_u1}})
    _, m21, _ = GoalDecompositionScorer().calculate_raw_metrics("s1", ctx_u1)
    assert m21["final_count"] == 1
    assert m21["instrumental_count"] == 0

    # 2_2 不触发实时 embed_texts，且纯计数唯一 c 节点 = 2
    raw_22, _, _ = InfoCollectionScorer().calculate_raw_metrics("s1", ctx)
    assert raw_22 == 2.0


def test_2_3_excludes_du_outdegree_and_3_2_excludes_quadratic_hops_and_3_3_zero_decision_gate():
    """2_3 仅计 info 预判不混入 d->u；3_2 不被最短路项淹没；3_3 0 决策严格得 0"""
    c1 = Node(id="c1", name="信息1", type="c")
    c2 = Node(id="c2", name="信息2", type="c")
    d1 = Node(id="d1", name="方案1", type="d")
    u1 = Node(id="u1", name="目标1", type="u")
    std = StandardGraph(
        corpus_id="g",
        nodes=[c1, c2, d1, u1],
        arcs=[
            Arc(source="d1", target="c1", type="causal"),
            Arc(source="d1", target="c2", type="causal"),
            Arc(source="d1", target="u1", type="causal"),
        ],
    )
    spk = _make_speaker("s1", ["发言"])
    transcript = DiscussionTranscript(group_id="g", speakers={"s1": spk}, utterances=spk.utterances)

    # 2_3: 只有 d1->u1 (无 info 边)，2_3 应为 0
    cg_du = CandidateGraph(
        speaker_id="s1",
        matched_nodes=[d1, u1],
        verified_arcs=[Arc(source="d1", target="u1", type="causal")],
    )
    ctx_du = ScoringContext(
        group_id="g", corpus_id="g", standard_graph=std, transcript=transcript,
        candidate_graphs={"s1": cg_du}, interaction_network=InteractionNetwork(),
    )
    raw_23, _, _ = ImpactPredictionScorer().calculate_raw_metrics("s1", ctx_du)
    assert raw_23 == 0.0

    # 3_2: 同轮提及两个无直连的发散信息叶子 c1, c2（即使未提枢轴 d1）应命中 1 次信息联想，且无最短路膨胀
    cg_leaves = CandidateGraph(
        speaker_id="s1",
        matched_nodes=[c1, c2],
        node_mentions=[
            NodeMention(node_id="c1", utterance_index=0, timestamp_seconds=0.0),
            NodeMention(node_id="c2", utterance_index=0, timestamp_seconds=0.0),
        ],
    )
    ctx_leaves = ctx_du.model_copy(update={"candidate_graphs": {"s1": cg_leaves}})
    raw_32, m32, _ = AssociationScorer().calculate_raw_metrics("s1", ctx_leaves)
    assert m32["info_associations"] == 1
    assert raw_32 == 1.0

    # 3_3: 0 决策节点（即使有 concepts 或 feasibility_ratings）必须严格归 0
    cg_zero_d = CandidateGraph(
        speaker_id="s1",
        matched_nodes=[c1, c2],
        feasibility_ratings={"resource_awareness": 9.0, "risk_sensitivity": 9.0, "execution_feasibility": 9.0, "adaptability": 9.0},
    )
    ctx_zero_d = ctx_du.model_copy(update={"candidate_graphs": {"s1": cg_zero_d}})
    raw_33, _, _ = FeasibilityScorer().calculate_raw_metrics("s1", ctx_zero_d)
    assert raw_33 == 0.0


def test_3_1_novel_nodes_not_triple_counted_and_4_includes_self_excitation_50_50():
    """3_1 创新节点不三重计数；4 思维激发恢复 50%/50% 自激+互激独立归一化"""
    d1 = Node(id="d1", name="降本方案", type="d")
    std = StandardGraph(corpus_id="g", nodes=[d1], arcs=[])
    s1 = _make_speaker("s1", ["提出降本方案", "继续深化降本方案落地细节"], times=[0.0, 20.0])
    s2 = _make_speaker("s2", ["补充客户需求视角", "推进降本方案试点"], times=[10.0, 30.0])
    transcript = DiscussionTranscript(group_id="g", speakers={"s1": s1, "s2": s2}, utterances=s1.utterances + s2.utterances)

    cg_novel = CandidateGraph(
        speaker_id="s1",
        matched_nodes=[],
        novel_nodes=[Node(id="nov_1", name="全新渠道模式", type="d")],
    )
    ctx = ScoringContext(
        group_id="g", corpus_id="g", standard_graph=std, transcript=transcript,
        candidate_graphs={"s1": cg_novel, "s2": CandidateGraph(speaker_id="s2", matched_nodes=[d1])},
        interaction_network=InteractionNetwork(speakers=["s1", "s2"], edges=[]),
    )
    raw_31, _, _ = CreativityScorer().calculate_raw_metrics("s1", ctx)
    # 单个 novel_node 只记一次适度加分 (1.5)，不再同时叠加 decision_count(1) + 3.5 = 4.5+
    assert raw_31 == pytest.approx(1.5)

    # 维度 4：raw_score 应落在 V5 的 [0.0, 2.0] (MinMax(self) + MinMax(peer)) 区间
    res_4 = InfluenceScorer().evaluate_all(ctx)
    for ds in res_4.values():
        assert 0.0 <= ds.raw_score <= 2.0


def test_5_1_5_2_5_3_negative_deny_never_boosts_openness_mediation_or_centrality():
    """P0-2/3/4 & 决策3：deny 绝不推高 5_1 开放性、5_2 冲突消解（带符号）和 5_3 协作中心性"""
    u0 = Utterance(index=0, speaker_id="s1", speaker_name="s1", text="提出初始方案", start_seconds=0.0)
    u1 = Utterance(index=1, speaker_id="s2", speaker_name="s2", text="反对1号方案，这完全不可行", start_seconds=20.0)
    u2 = Utterance(index=2, speaker_id="s3", speaker_name="s3", text="我理解双方顾虑，赞同先折中试点", start_seconds=40.0)
    u3 = Utterance(index=3, speaker_id="s2", speaker_name="s2", text="我依然强烈反对3号和1号", start_seconds=60.0)

    spks = {
        "s1": Speaker(speaker_id="s1", display_name="s1", utterances=[u0], turn_count=1),
        "s2": Speaker(speaker_id="s2", display_name="s2", utterances=[u1, u3], turn_count=2),
        "s3": Speaker(speaker_id="s3", display_name="s3", utterances=[u2], turn_count=1),
    }
    transcript = DiscussionTranscript(group_id="g", speakers=spks, utterances=[u0, u1, u2, u3], total_duration_seconds=80.0)
    net = InteractionNetwork(
        speakers=["s1", "s2", "s3"],
        edges=[
            InteractionEdge(source="s2", target="s1", type="deny", weight=1.0, utterance_index=1, timestamp_seconds=20.0),
            InteractionEdge(source="s3", target="s1", type="support", weight=1.0, utterance_index=2, timestamp_seconds=40.0),
            InteractionEdge(source="s2", target="s3", type="deny", weight=1.0, utterance_index=3, timestamp_seconds=60.0),
        ],
    )
    ctx = ScoringContext(
        group_id="g",
        corpus_id="g",
        standard_graph=StandardGraph(corpus_id="g", nodes=[], arcs=[]),
        transcript=transcript,
        candidate_graphs={"s1": CandidateGraph(speaker_id="s1"), "s2": CandidateGraph(speaker_id="s2"), "s3": CandidateGraph(speaker_id="s3")},
        interaction_network=net,
    )

    # 5_1: s2 只发出 deny，开放性 raw 必须为 0，且严格低于发出 support 的 s3
    r51_s2, m51_s2, _ = OpennessScorer().calculate_raw_metrics("s2", ctx)
    r51_s3, _, _ = OpennessScorer().calculate_raw_metrics("s3", ctx)
    assert r51_s2 == 0.0
    assert m51_s2["interacted_peers"] == 0
    assert r51_s3 > r51_s2

    # 5_2: s3 冲突后正向调解 (>0)，s2 持续 deny 升级冲突 (<0)
    r52_s3, _, _ = ConflictResolutionScorer().calculate_raw_metrics("s3", ctx)
    r52_s2, _, _ = ConflictResolutionScorer().calculate_raw_metrics("s2", ctx)
    assert r52_s3 > 0.0
    assert r52_s2 < 0.0

    # 5_3: s3 被 deny (净入度 -1)，s1 有 1 support + 1 deny (净入度 0)，s2 净入度 0；
    # 冲突枢纽 s2 不得靠无向 betweenness 反超无净负入度的 s1
    r53_s1, _, _ = CollaborationCentralityScorer().calculate_raw_metrics("s1", ctx)
    r53_s2, _, _ = CollaborationCentralityScorer().calculate_raw_metrics("s2", ctx)
    r53_s3, _, _ = CollaborationCentralityScorer().calculate_raw_metrics("s3", ctx)
    assert r53_s1 == pytest.approx(0.0)
    assert r53_s2 == pytest.approx(0.0)
    assert r53_s3 < 0.0


def test_6_1_pure_sim_and_6_2_tight_regex_timing_dominance_and_6_3_delta_dominance():
    """6_1 不受 matched_ratio 干扰；6_2 不误判'通勤30分钟'且控时主导；6_3 total_delta 不被后半程发言量淹没"""
    # 6_2: s1 说“通勤30分钟”且 3 阶段均衡发言，s2 明确提醒“还剩5分钟，抓紧时间进入下一阶段”仅 1 阶段
    s1 = _make_speaker("s1", ["用户每天通勤30分钟", "产品充电5分钟", "总体成本偏高"], times=[10.0, 50.0, 90.0])
    s2 = _make_speaker("s2", ["大家注意时间，还剩5分钟我们抓紧进入下一阶段总结"], times=[85.0])
    transcript = DiscussionTranscript(
        group_id="g",
        core_topic="降低运营成本并提升客户留存",
        convergence_decision="聚焦核心产品降本与客户留存方案",
        speakers={"s1": s1, "s2": s2},
        utterances=s1.utterances + s2.utterances,
        total_duration_seconds=100.0,
    )
    std = StandardGraph(corpus_id="g", nodes=[Node(id=f"c{i}", name=f"节点{i}", type="c") for i in range(5)], arcs=[])
    # s1 命中全部 5 个节点，s2 命中 0 个，但 6_1 只看 mean_sim
    cg1 = CandidateGraph(
        speaker_id="s1",
        matched_nodes=list(std.nodes),
        node_mentions=[NodeMention(node_id=f"c{i}", utterance_index=2, timestamp_seconds=90.0) for i in range(5)],
    )
    cg2 = CandidateGraph(speaker_id="s2", matched_nodes=[])
    ctx = ScoringContext(
        group_id="g", corpus_id="g", standard_graph=std, transcript=transcript,
        candidate_graphs={"s1": cg1, "s2": cg2}, interaction_network=InteractionNetwork(),
    )

    # 6_2: s1 控时次数应为 0（不误命中“通勤30分钟”“充电5分钟”），且 s2 > s1
    r62_s1, m62_s1, _ = TimingControlScorer().calculate_raw_metrics("s1", ctx)
    r62_s2, m62_s2, _ = TimingControlScorer().calculate_raw_metrics("s2", ctx)
    assert m62_s1["timing_control_count"] == 0
    assert m62_s2["timing_control_count"] >= 1
    assert r62_s2 > r62_s1

    # 6_1: raw 应等于 topic_cosine_similarity（不被 matched_ratio*5.0 抬升到 >1.0）
    r61_s1, m61_s1, _ = FocusScorer().calculate_raw_metrics("s1", ctx)
    assert r61_s1 == pytest.approx(m61_s1["topic_cosine_similarity"], abs=1e-3)

    # 6_3: raw 应直接等于 v5_convergence_delta（不被 5 个 late_mentions*1.5=+7.5 淹没）
    r63_s1, m63_s1, _ = ConvergenceScorer().calculate_raw_metrics("s1", ctx)
    assert r63_s1 == pytest.approx(m63_s1["v5_convergence_delta"], abs=1e-3)
