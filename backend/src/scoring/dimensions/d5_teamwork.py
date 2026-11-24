# -*- coding: utf-8 -*-
"""
Dimension 5: Conflict Resolution & Teamwork Competence Scorers (冲突协调能力 - 对齐 V5 团队动力模型)
- 5_1: Openness (开放性: 积极倾听+认可他人+分析分歧+推动共识 四维计数与互动广度)
- 5_2: Conflict Resolution (冲突消解: 分歧发生后的时序介入调解、折中方案供给与观点收敛)
- 5_3: Collaboration Centrality (协作中心性: 对齐 V5 CollaborationNetwork 有向带符号中心性 + 中介中心性)
"""

from typing import Any, Dict, List

import networkx as nx

from src.scoring.base import BaseDimensionScorer, ScoringContext


class OpennessScorer(BaseDimensionScorer):
    """5_1 开放度评分器 (严格对齐 V5: listening + supporting + analysis + pushing，排除 deny 边)"""

    SCORER_ID = "5_1"
    SCORER_NAME = "冲突协调能力 - 开放性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        out_edges = [e for e in context.interaction_network.edges if e.source == speaker_id]
        support_out = [e for e in out_edges if e.type == "support"]
        neutral_out = [e for e in out_edges if e.type == "neutral"]
        # 仅统计正向/中立互动对象，绝不把 deny 攻击对象计入开放性覆盖（修复 P0-2）
        targeted_speakers = {e.target for e in support_out + neutral_out}

        cand_graph = context.get_candidate_graph(speaker_id)
        oc = (cand_graph.openness_counts if cand_graph else None) or {}

        listening = oc.get("listening", len(neutral_out))
        supporting = oc.get("supporting", len(support_out))
        analysis = oc.get("analysis", 0)
        pushing = oc.get("pushing", 0)
        v5_openness_sum = listening + supporting + analysis + pushing

        # 严格对齐 V5 公式: 纯四维开放行为之和（消除 supporting 双重计数与 deny 误加分）
        total_raw = float(v5_openness_sum)

        key_points = []
        if supporting > 0 or len(support_out) > 0:
            cnt = max(supporting, len(support_out))
            key_points.append(f"尊重并正向接纳他人合理见解 {cnt} 次，展现出良好的开放协作姿态")
        if len(targeted_speakers) >= 2:
            key_points.append(f"主动正向回应覆盖 {len(targeted_speakers)} 位不同成员，倾听与互动面广")
        elif v5_openness_sum > 0:
            key_points.append(f"累计展现 {v5_openness_sum} 次倾听反馈与共识推进动作")

        metrics = {
            "listening_count": listening,
            "supporting_count": supporting,
            "conflict_analysis_count": analysis,
            "consensus_pushing_count": pushing,
            "interacted_peers": len(targeted_speakers),
        }
        return float(total_raw), metrics, key_points


class ConflictResolutionScorer(BaseDimensionScorer):
    """5_2 冲突消解能力评分器 (带正负号的轻量冲突调解分：调和/支持加分，激化/反驳扣分)"""

    SCORER_ID = "5_2"
    SCORER_NAME = "冲突协调能力 - 冲突消解"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        network_edges = context.interaction_network.edges
        deny_edges = [e for e in network_edges if e.type == "deny"]

        conflict_turn_indices = [e.utterance_index for e in deny_edges if e.utterance_index is not None]
        cand_graph = context.get_candidate_graph(speaker_id)
        speaker = context.get_speaker(speaker_id)

        my_out = [e for e in network_edges if e.source == speaker_id]
        my_deny = [e for e in my_out if e.type == "deny"]
        my_support = [e for e in my_out if e.type == "support"]

        # 按发言轮次索引本人的互动极性，严格区分冲突后是“调解”还是“激化升级”（修复 P0-3）
        my_types_by_turn: Dict[int, set[str]] = {}
        for e in my_out:
            if e.utterance_index is not None:
                my_types_by_turn.setdefault(e.utterance_index, set()).add(e.type)

        constructive_mediations = 0
        conflict_escalations = 0
        if speaker and conflict_turn_indices:
            for u in speaker.utterances:
                if any(0 < (u.index - c_idx) <= 3 for c_idx in conflict_turn_indices):
                    turn_types = my_types_by_turn.get(u.index, set())
                    if "deny" in turn_types:
                        conflict_escalations += 1
                    else:
                        constructive_mediations += 1

        decision_nodes = (
            [n for n in cand_graph.matched_nodes if n.type == "d"]
            if cand_graph
            else []
        )
        conflict_analysis_cnt = (
            cand_graph.openness_counts.get("analysis", 0) if cand_graph and cand_graph.openness_counts else 0
        )

        # 带正负号的冲突消解分：正向调解/分析/支持加分，持续反驳与冲突升级明确扣分
        total_raw = (
            constructive_mediations * 2.5
            + conflict_analysis_cnt * 1.5
            + len(my_support) * 1.0
            - conflict_escalations * 2.5
            - len(my_deny) * 1.0
        )

        key_points = []
        if constructive_mediations > 0:
            key_points.append(f"在团队出现观点交锋后及时建设性介入调和 {constructive_mediations} 次，缓和场上对立")
        if len(my_support) > 0 and len(decision_nodes) > 0:
            key_points.append(f"结合 {len(decision_nodes)} 项建设性折中方案凝聚成员共识")
        elif len(my_support) > 0:
            key_points.append(f"通过 {len(my_support)} 次正向整合回应化解潜在分歧")

        metrics = {
            "post_conflict_mediations": constructive_mediations,
            "post_conflict_escalations": conflict_escalations,
            "conflict_analysis_count": conflict_analysis_cnt,
            "alignment_support": len(my_support),
            "deny_count": len(my_deny),
            "consensus_decisions": len(decision_nodes),
        }
        return float(total_raw), metrics, key_points


class CollaborationCentralityScorer(BaseDimensionScorer):
    """5_3 协作中心性评分器 (严格对齐 V5 CollaborationNetwork: support(+w)/deny(-w) 有向带符号入度中心性)"""

    SCORER_ID = "5_3"
    SCORER_NAME = "冲突协调能力 - 协作中心性"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        G = context.interaction_network.to_networkx()

        if G.number_of_nodes() == 0 or not G.has_node(speaker_id):
            return 0.0, {"net_in_degree_weight": 0.0, "v5_collab_centrality": 0.0, "betweenness_centrality": 0.0}, []

        # 严格对齐 V5: 仅对 support (+w) 与 deny (-w) 计算带符号入度中心性，排除 neutral 与无符号 betweenness 干扰（修复 P0-4）
        signed_edges = [e for e in context.interaction_network.edges if e.type in ("support", "deny")]
        in_signed_weight = 0.0
        for e in signed_edges:
            if e.target == speaker_id:
                in_signed_weight += float(e.weight) if e.type == "support" else -float(e.weight)

        total_signed_edges = len(signed_edges)
        v5_centrality = in_signed_weight / total_signed_edges if total_signed_edges > 0 else 0.0

        # 中介中心性仅保留在 metrics 中供可视化/诊断参考，不再奖励冲突枢纽
        try:
            betweenness = nx.betweenness_centrality(G.to_undirected()).get(speaker_id, 0.0)
        except Exception:
            betweenness = 0.0

        total_raw = float(v5_centrality)

        key_points = []
        if in_signed_weight >= 1.5:
            key_points.append(f"处于团队协作网络核心位置（协作中心度 {v5_centrality:.2f}），获高频正向响应")
        if not key_points and in_signed_weight > 0:
            key_points.append("在团队协作网络中获得正向支持与认可")

        metrics = {
            "v5_collab_centrality": round(v5_centrality, 3),
            "net_in_degree_weight": round(in_signed_weight, 2),
            "betweenness_centrality": round(betweenness, 3),
        }
        return float(total_raw), metrics, key_points
