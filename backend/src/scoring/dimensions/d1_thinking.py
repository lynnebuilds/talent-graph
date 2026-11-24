# -*- coding: utf-8 -*-
"""
Dimension 1: Thinking Competence Scorers (思维能力 - 100% 对齐 V5 影响图推理与金字塔理论)
- 1_1: Logical Reasoning (逻辑思维: 概念↔事实 演绎/归纳/溯因推理连通簇 + 决策链)
- 1_2: Abstraction & Induction (抽象归纳: 同属上位概念且在因果图中 d-分离的跨事实归纳对)
- 1_3: Systematic Thinking (系统思维: 宏观主题[一级] × 抽象概念[二级] × 具体事实[三级] 金字塔乘积)
"""

from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from src.scoring.base import BaseDimensionScorer, ScoringContext


def _count_connected_components(edges: List[Tuple[int, int]]) -> int:
    """无向连通分量计数（对齐 V5 find_connected_components_from_edges，按发言段内位置索引识别独立推理簇）"""
    if not edges:
        return 0
    adj: Dict[int, Set[int]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    visited: Set[int] = set()
    comp = 0
    for node in adj:
        if node in visited:
            continue
        comp += 1
        stack = [node]
        visited.add(node)
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
    return comp


class LogicalReasoningScorer(BaseDimensionScorer):
    """1_1 逻辑思维能力评分器 (严格对齐 V5: 按发言段位置图计算演绎/归纳/溯因三模式连通簇)"""

    SCORER_ID = "1_1"
    SCORER_NAME = "思维能力 - 逻辑思维"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"deduction_count": 0, "induction_count": 0, "abduction_count": 0, "chain_count": 0}, []

        std_graph = context.standard_graph
        has_levels = std_graph.has_explicit_levels

        if has_levels:
            concept_ids = std_graph.concept_node_ids
            fact_ids = std_graph.fact_node_ids
        else:
            # 兼容未标注层级的极简测试图谱：以 u/d 作为上位概念层、c 作为基础事实层
            concept_ids = {n.id for n in std_graph.nodes if n.type in ("u", "d")}
            fact_ids = {n.id for n in std_graph.nodes if n.type == "c"}

        # 对齐 V5: 仅使用候选人已验证的 causal 边（避免 0 边回退全图导致从 0->1 非单调骤降）
        active_edges: Set[Tuple[str, str]] = {
            (a.source, a.target) for a in cand_graph.verified_arcs if a.type == "causal"
        }

        # 按发言轮次(utterance_index)分组（对齐 V5 逐段落检测，禁止跨轮次拼接假推理边）
        utterance_seqs: List[List[str]] = []
        if cand_graph.node_mentions:
            by_utt: Dict[int, List[str]] = {}
            for m in sorted(cand_graph.node_mentions, key=lambda x: (x.utterance_index, x.timestamp_seconds)):
                by_utt.setdefault(m.utterance_index, []).append(m.node_id)
            utterance_seqs = list(by_utt.values())
        elif cand_graph.matched_nodes:
            utterance_seqs = [[n.id for n in cand_graph.matched_nodes]]

        deduction_count = 0
        induction_count = 0
        abduction_count = 0

        for seq in utterance_seqs:
            n_seq = len(seq)
            if n_seq < 2:
                continue

            # Pattern 1 演绎推理 (V5 双遍法):
            # Pass 1: C -> F 边，将目标事实位置标记为已由概念推导 (flag[j] = 1)
            # Pass 2: 仅当前置事实已被概念推导 (flag[i] == 1) 时，才允许串联 F -> F 演绎边
            ded_edges: List[Tuple[int, int]] = []
            deduced_flag = [0] * n_seq
            for i in range(n_seq):
                u_id = seq[i]
                if u_id not in concept_ids:
                    continue
                for j in range(i + 1, n_seq):
                    v_id = seq[j]
                    if u_id != v_id and v_id in fact_ids and (u_id, v_id) in active_edges:
                        ded_edges.append((i + 1, j + 1))
                        deduced_flag[j] = 1
            for i in range(n_seq):
                u_id = seq[i]
                if u_id not in fact_ids or deduced_flag[i] != 1:
                    continue
                for j in range(i + 1, n_seq):
                    v_id = seq[j]
                    if u_id != v_id and v_id in fact_ids and (u_id, v_id) in active_edges:
                        ded_edges.append((i + 1, j + 1))
                        deduced_flag[j] = 1

            # Pattern 3 归纳推理: 先提事实 F 后归纳概念 C (或 C -> C)
            ind_edges: List[Tuple[int, int]] = []
            for i in range(n_seq):
                u_id = seq[i]
                for j in range(i + 1, n_seq):
                    v_id = seq[j]
                    if u_id != v_id and (u_id, v_id) in active_edges:
                        if (u_id in fact_ids and v_id in concept_ids) or (u_id in concept_ids and v_id in concept_ids):
                            ind_edges.append((i + 1, j + 1))

            # Pattern 4 溯因推理: 先提结果事实 u_id，后追溯原因 v_id (存在 v_id -> u_id)
            abd_edges: List[Tuple[int, int]] = []
            for i in range(n_seq):
                u_id = seq[i]
                if u_id not in fact_ids:
                    continue
                for j in range(i + 1, n_seq):
                    v_id = seq[j]
                    if u_id != v_id and (v_id, u_id) in active_edges:
                        abd_edges.append((j + 1, i + 1))

            deduction_count += _count_connected_components(ded_edges)
            induction_count += _count_connected_components(ind_edges)
            abduction_count += _count_connected_components(abd_edges)

        # 多跳深度因果链计数 (保留作观测指标，不干扰 V5 核心公式)
        G = cand_graph.to_networkx()
        chain_count = 0
        for src in G.nodes():
            for tgt in G.nodes():
                if src != tgt and nx.has_path(G, src, tgt):
                    try:
                        for p in nx.all_simple_paths(G, src, tgt, cutoff=3):
                            if len(p) >= 3:
                                chain_count += 1
                    except Exception:
                        pass

        # 严格对齐 V5 主公式: count_deduction + count_induction + count_abduction
        total_raw = float(deduction_count + induction_count + abduction_count)

        key_points = []
        if deduction_count > 0:
            key_points.append(f"构建了 {deduction_count} 个演绎推理簇（由概念/依据推演具体结论）")
        if induction_count > 0:
            key_points.append(f"完成了 {induction_count} 个归纳推理簇（由具体事实提炼上位判断）")
        if abduction_count > 0:
            key_points.append(f"体现了 {abduction_count} 次由果溯因的逆向溯因分析")
        if chain_count > 0 and len(key_points) < 2:
            key_points.append(f"形成 {chain_count} 条多跳深度因果链条")
        if not key_points and total_raw > 0:
            key_points.append("展现了一定的逻辑论证与因果关联能力")

        metrics = {
            "deduction_count": deduction_count,
            "induction_count": induction_count,
            "abduction_count": abduction_count,
            "chain_count": chain_count,
            "verified_arcs": len(cand_graph.verified_arcs),
        }
        return float(total_raw), metrics, key_points


class AbstractionScorer(BaseDimensionScorer):
    """1_2 抽象归纳能力评分器 (严格对齐 V5: 逐发言段统计同上位概念且在因果图中不直接相连的具体节点对)"""

    SCORER_ID = "1_2"
    SCORER_NAME = "思维能力 - 抽象归纳"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"abstraction_count": 0, "concept_nodes_count": 0, "concept_coverage": 0.0}, []

        std_graph = context.standard_graph
        causal_undirected: Set[Tuple[str, str]] = set()
        causal_nodes: Set[str] = set()
        for a in std_graph.arcs:
            if a.type == "causal":
                causal_undirected.add((a.source, a.target))
                causal_undirected.add((a.target, a.source))
                causal_nodes.add(a.source)
                causal_nodes.add(a.target)

        # 按发言轮次分组（对齐 V5 逐段落统计；若无 mention 退化为单段 matched_nodes）
        utterance_nodes: List[List[Any]] = []
        if cand_graph.node_mentions:
            by_utt: Dict[int, List[Any]] = {}
            for m in sorted(cand_graph.node_mentions, key=lambda x: (x.utterance_index, x.timestamp_seconds)):
                nd = std_graph.get_node(m.node_id)
                if nd is not None:
                    by_utt.setdefault(m.utterance_index, []).append(nd)
            utterance_nodes = list(by_utt.values())
        elif cand_graph.matched_nodes:
            utterance_nodes = [list(cand_graph.matched_nodes)]

        # V5 核心指标：同段内提及两个同属同一上位 parent_concept、属于因果图节点且无直接因果边的不同具体节点对
        v5_abstraction_pairs: List[Tuple[str, str]] = []
        for seg_nodes in utterance_nodes:
            n_len = len(seg_nodes)
            for i in range(n_len):
                ni = seg_nodes[i]
                ci = ni.parent_concept
                if not ci or ni.name == ci or ni.id == ci or ni.id not in causal_nodes:
                    continue
                for j in range(i + 1, n_len):
                    nj = seg_nodes[j]
                    cj = nj.parent_concept
                    if ci != cj or ni.id == nj.id or nj.name == cj or nj.id == cj or nj.id not in causal_nodes:
                        continue
                    if (ni.id, nj.id) not in causal_undirected:
                        v5_abstraction_pairs.append((ni.name, nj.name))

        matched_nodes = cand_graph.matched_nodes
        if std_graph.has_explicit_levels:
            concept_nodes = [n for n in matched_nodes if n.level in ("一级", "二级")]
            total_std_concepts = max(1, len(std_graph.concept_node_ids))
        else:
            concept_nodes = [n for n in matched_nodes if n.type == "c"]
            total_std_concepts = max(1, len([n for n in std_graph.nodes if n.type == "c"]))

        concept_coverage = len(concept_nodes) / total_std_concepts
        abstraction_count = len(v5_abstraction_pairs)
        # 严格对齐 V5 公式: score = len(v5_abstraction_pairs)
        total_raw = float(abstraction_count)

        key_points = []
        if v5_abstraction_pairs:
            p0 = v5_abstraction_pairs[0]
            key_points.append(
                f"完成 {len(v5_abstraction_pairs)} 组跨独立事实的同概念抽象提炼（如「{p0[0][:10]}」与「{p0[1][:10]}」）"
            )
        if len(concept_nodes) >= 2:
            key_points.append(f"准确把握 {len(concept_nodes)} 项核心抽象概念，概念覆盖率 {concept_coverage:.0%}")

        metrics = {
            "abstraction_count": abstraction_count,
            "v5_concept_dsep_pairs": len(v5_abstraction_pairs),
            "concept_nodes_count": len(concept_nodes),
            "concept_coverage": round(concept_coverage, 2),
        }
        return float(total_raw), metrics, key_points


class SystematicThinkingScorer(BaseDimensionScorer):
    """1_3 系统思维能力评分器 (严格对齐 V5 公式: (1 + 一级宏观主题) × 二级抽象概念 × 三级具体事实)"""

    SCORER_ID = "1_3"
    SCORER_NAME = "思维能力 - 系统思维"

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        cand_graph = context.get_candidate_graph(speaker_id)
        if not cand_graph:
            return 0.0, {"level1_count": 0, "level2_count": 0, "level3_count": 0, "pyramid_product": 0}, []

        # 显式按节点 ID 去重且仅统计标准节点（对齐 V5 node_output_dropdup）
        uniq_std_nodes = list({n.id: n for n in cand_graph.matched_nodes}.values())
        std_graph = context.standard_graph
        if std_graph.has_explicit_levels:
            level1 = sum(1 for n in uniq_std_nodes if n.level == "一级")
            level2 = sum(1 for n in uniq_std_nodes if n.level == "二级")
            level3 = sum(1 for n in uniq_std_nodes if n.level == "三级")
        else:
            # 极简无层级测试夹具回退：按 u(宏观目标) / d(中观举措) / c(微观事实) 统计
            level1 = sum(1 for n in uniq_std_nodes if n.type == "u")
            level2 = sum(1 for n in uniq_std_nodes if n.type == "d")
            level3 = sum(1 for n in uniq_std_nodes if n.type == "c")

        # 严格对齐 V5 原始公式：(1 + 一级) × 二级 × 三级（移除奖励极小图的 density 与非 V5 线性覆盖项）
        v5_pyramid = (1 + level1) * level2 * level3
        G = cand_graph.to_networkx()
        density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
        total_raw = float(v5_pyramid)

        key_points = []
        if level1 > 0 and level2 > 0 and level3 > 0:
            key_points.append(
                f"兼备一级主题({level1})、二级概念({level2})与三级事实({level3})的完整三层金字塔思维结构"
            )
        elif level2 > 0 and level3 > 0:
            key_points.append(f"较好串联了中观抽象概念({level2})与微观具体事实({level3})")
        if len(cand_graph.verified_arcs) >= 3:
            key_points.append(f"构建了含 {len(cand_graph.verified_arcs)} 条验证关系的系统关联子网")

        metrics = {
            "level1_count": level1,
            "level2_count": level2,
            "level3_count": level3,
            "pyramid_product": v5_pyramid,
            "graph_density": round(density, 3),
        }
        return float(total_raw), metrics, key_points
