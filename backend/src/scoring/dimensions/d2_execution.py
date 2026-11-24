# -*- coding: utf-8 -*-
"""
Dimension 2: Execution & Decision-Making Competence Scorers (决策能力 - 100% 对齐 V5 决策分析理论)
- 2_1: Goal Identification & Decomposition (目标识别与拆解: (手段目标+最终目标)×(1+d->手段->最终拆解路径))
- 2_2: Information Collection & Integration (信息收集与整合: 已知/未知信息广度 + 向量空间覆盖分散度)
- 2_3: Impact Prediction (方案影响预判: 信息支撑决策 c->d 与决策反作用于环境 d->c 的双向影响预判)
"""

from typing import Any, Dict, List, Set, Tuple

from src.scoring.base import BaseDimensionScorer, ScoringContext


class GoalDecompositionScorer(BaseDimensionScorer):
    """2_1 目标识别与拆解能力评分器 (严格对齐 V5: (手段目标+最终目标) × (1 + 逐段拆解子图命中数))"""

    SCORER_ID = "2_1"
    SCORER_NAME = "决策能力 - 目标识别与拆解"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"final_count": 0, "instrumental_count": 0, "decompose_count": 0}, []

        std_graph = context.standard_graph
        final_ids = set(std_graph.final_target_ids)
        inst_ids = set(std_graph.instrumental_target_ids)

        # 仅当 final 与 instrumental 均未显式标注时，按 u->u 拓扑自动推断（修复 P0-5 and/or 覆盖 Bug）
        std_u_ids = {n.id for n in std_graph.nodes if n.type == "u"}
        if std_u_ids and not (final_ids or inst_ids):
            u_with_out_u = {
                a.source for a in std_graph.arcs if a.source in std_u_ids and a.target in std_u_ids
            }
            if u_with_out_u:
                inst_ids = u_with_out_u
                final_ids = std_u_ids - u_with_out_u
            else:
                final_ids = std_u_ids

        # 构建 V5 目标拆解三元组: d -> u_inst -> u_final (且不存在 d -> u_final 直接跳边)
        d_ids_std = {n.id for n in std_graph.nodes if n.type == "d"}
        d_u_edges = {(a.source, a.target) for a in std_graph.arcs if a.source in d_ids_std and a.target in std_u_ids}
        u_u_edges = {(a.source, a.target) for a in std_graph.arcs if a.source in std_u_ids and a.target in std_u_ids}

        v5_subgraphs: List[Tuple[str, str, str]] = []
        for d_id, u_mid in d_u_edges:
            if u_mid in inst_ids:
                for u1, u_fin in u_u_edges:
                    if u1 == u_mid and u_fin in final_ids and (d_id, u_fin) not in d_u_edges:
                        v5_subgraphs.append((d_id, u_mid, u_fin))

        matched_set = cand_graph.matched_node_ids
        final_matched = [nid for nid in matched_set if nid in final_ids]
        inst_matched = [nid for nid in matched_set if nid in inst_ids]

        # 对齐 V5: 逐发言段统计 (d, u_fin) 同段共现命中数（无 mention 时回退为单段 matched_set）
        utt_node_sets: List[Set[str]] = []
        if cand_graph.node_mentions:
            by_utt: Dict[int, Set[str]] = {}
            for m in cand_graph.node_mentions:
                by_utt.setdefault(m.utterance_index, set()).add(m.node_id)
            utt_node_sets = list(by_utt.values())
        elif matched_set:
            utt_node_sets = [matched_set]

        decompose_count = 0
        for u_nodes in utt_node_sets:
            decompose_count += sum(
                1 for (d_id, _, u_fin) in v5_subgraphs if d_id in u_nodes and u_fin in u_nodes
            )

        cand_d_ids = {n.id for n in cand_graph.matched_nodes if n.type == "d"}
        # 严格对齐 V5 核心公式：(手段目标数 + 最终目标数) × (1 + 拆解次数)；0 目标严格为 0
        v5_score = (len(inst_matched) + len(final_matched)) * (1 + decompose_count)
        total_raw = float(v5_score)

        key_points = []
        if final_matched or inst_matched:
            key_points.append(
                f"识别了 {len(final_matched)} 项最终目标与 {len(inst_matched)} 项阶段性手段目标"
            )
        if decompose_count > 0:
            key_points.append(f"完成了 {decompose_count} 条「方案举措 → 手段目标 → 最终目标」拆解链路")

        metrics = {
            "final_count": len(final_matched),
            "instrumental_count": len(inst_matched),
            "decompose_count": decompose_count,
            "decision_count": len(cand_d_ids),
        }
        return float(total_raw), metrics, key_points


class InfoCollectionScorer(BaseDimensionScorer):
    """2_2 信息收集与整合能力评分器 (严格对齐 V5: 唯一信息节点 c 计数，零实时向量调用/100%纯内存确定性)"""

    SCORER_ID = "2_2"
    SCORER_NAME = "决策能力 - 信息收集与整合"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"concept_nodes": 0, "unknown_explored": 0, "total_mentions": 0}, []

        c_nodes = list({n.id: n for n in cand_graph.matched_nodes if n.type == "c"}.values())
        unknown_c = [n for n in c_nodes if n.sub_type == "unknown"]
        c_ids = {n.id for n in c_nodes}
        c_mentions = [m for m in cand_graph.node_mentions if m.node_id in c_ids]

        time_span = 0.0
        if c_mentions:
            ts_list = [m.timestamp_seconds for m in c_mentions]
            time_span = max(ts_list) - min(ts_list)

        # 严格对齐 V5 s2_2_info_collection: score = len(unique c nodes)（移除实时 embed_texts 调用，守护纯内存红线）
        total_raw = float(len(c_nodes))

        key_points = []
        if len(c_nodes) >= 2:
            key_points.append(f"整合引用了 {len(c_nodes)} 项客观事实与背景依据（含 {len(unknown_c)} 项深层待查证信息）")
        if len(c_mentions) >= 3:
            key_points.append(f"全场高频调用事实论据 {len(c_mentions)} 次支撑观点")

        metrics = {
            "concept_nodes": len(c_nodes),
            "unknown_explored": len(unknown_c),
            "total_mentions": len(c_mentions),
            "time_span_seconds": round(time_span, 1),
        }
        return float(total_raw), metrics, key_points


class ImpactPredictionScorer(BaseDimensionScorer):
    """2_3 方案影响预判能力评分器 (严格对齐 V5: 逐段统计 info_arcs 命中数，不混入 d->u 边)"""

    SCORER_ID = "2_3"
    SCORER_NAME = "决策能力 - 方案影响预判"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"impact_count": 0, "feedback_loops": 0, "decision_count": 0}, []

        std_graph = context.standard_graph
        node_type_map = {n.id: n.type for n in std_graph.nodes}

        # 对齐 V5 s2_3_impact_prediction: info_arcs 集合（同时统计 d->c 反向反馈作为观测指标，避免重复计数与 d->u 污染）
        info_arcs_set: Set[Tuple[str, str]] = {(a.source, a.target) for a in std_graph.arcs if a.type == "info"}
        causal_dc_set: Set[Tuple[str, str]] = {
            (a.source, a.target)
            for a in std_graph.arcs
            if a.type == "causal" and node_type_map.get(a.source) == "d" and node_type_map.get(a.target) == "c"
        }

        utt_node_sets: List[Set[str]] = []
        if cand_graph.node_mentions:
            by_utt: Dict[int, Set[str]] = {}
            for m in cand_graph.node_mentions:
                by_utt.setdefault(m.utterance_index, set()).add(m.node_id)
            utt_node_sets = list(by_utt.values())
        elif cand_graph.matched_node_ids:
            utt_node_sets = [cand_graph.matched_node_ids]

        v5_impact_hits = 0
        feedback_hits = 0
        for u_nodes in utt_node_sets:
            v5_impact_hits += sum(1 for (src, tgt) in info_arcs_set if src in u_nodes and tgt in u_nodes)
            feedback_hits += sum(
                1
                for (d_id, c_id) in causal_dc_set
                if d_id in u_nodes and c_id in u_nodes and (c_id, d_id) in info_arcs_set
            )

        decision_ids = [n.id for n in cand_graph.matched_nodes if n.type == "d"]
        # 严格对齐 V5 公式: score = v5_impact_hits（不再叠加 d->u 出度与裸决策数）
        total_raw = float(v5_impact_hits)

        key_points = []
        if feedback_hits > 0:
            key_points.append(f"针对决策方案推演了 {feedback_hits} 项双向环境反馈效应")
        if v5_impact_hits > 0:
            key_points.append(f"准确建立 {v5_impact_hits} 组「客观条件 ↔ 方案选择」影响预判")
        if not key_points and total_raw > 0:
            key_points.append("具备基础的方案落地影响评估意识")

        metrics = {
            "impact_count": v5_impact_hits,
            "feedback_loops": feedback_hits,
            "decision_count": len(decision_ids),
        }
        return float(total_raw), metrics, key_points
