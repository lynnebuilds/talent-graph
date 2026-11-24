# -*- coding: utf-8 -*-
"""
Dimension 4: Thought Stimulation & Influence Competence Scorer (思维激发能力)
Restores V5's signature Multivariate Hawkes Point Process (多变量语义耦合霍克斯自激/互激点过程)
via pure-NumPy vectorized Minorize-Maximize (EM) maximum likelihood estimation (< 2ms in memory!).
"""

from typing import Any, Dict, List, Set

import numpy as np

from src.core.embedding import get_embedding_engine
from src.models.discussion import Utterance
from src.models.score import DimensionScore
from src.scoring.base import BaseDimensionScorer, ScoringContext, normalize_to_standard_score


class InfluenceScorer(BaseDimensionScorer):
    """
    4 思维激发能力评分器 (还原 V5 多变量语义 Hawkes 点过程互激强度估计 α_{sender -> receiver})
    对讨论事件流 {(t_m, speaker_m, vec_m, nodes_m)} 建立强度函数:
        λ_v(t) = μ_v + ∑_{t_i < t} α_{s_i -> v} · W_sem(i, j) · exp(-β (t - t_i))
    采用全向量化 EM 算法求解自激项 α_self (V5 alpha_sbase) 与跨人思维激发项 α_peer (V5 alpha_obase)。
    """

    SCORER_ID = "4"
    SCORER_NAME = "思维激发能力"

    def _collect_chronological_events(self, context: ScoringContext) -> List[Utterance]:
        cand_set = set(context.candidate_ids)
        from_transcript = [u for u in context.transcript.utterances if u.speaker_id in cand_set]
        from_speakers: List[Utterance] = []
        for cid in context.candidate_ids:
            spk = context.get_speaker(cid)
            if spk and spk.utterances:
                from_speakers.extend(spk.utterances)

        events = from_transcript if len(from_transcript) >= len(from_speakers) else from_speakers
        return sorted(events, key=lambda u: (u.start_seconds, u.index))

    def _fit_multivariate_hawkes(
        self, context: ScoringContext
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        全向量化多变量语义耦合 Hawkes 过程 EM 估计：
        返回 (peer_influence_alpha, self_excitation_alpha, node_adoptions_map)
        """
        cand_ids = context.candidate_ids
        peer_alpha_map = {cid: 0.0 for cid in cand_ids}
        self_alpha_map = {cid: 0.0 for cid in cand_ids}
        adoptions_map = {cid: 0 for cid in cand_ids}

        if len(cand_ids) < 2:
            return peer_alpha_map, self_alpha_map, adoptions_map

        # 统计率先提出节点被他人后续采纳的次数（使用 (timestamp_seconds, utterance_index) 兼容零时间戳文本）
        first_mention_owner: Dict[str, tuple[str, tuple[float, int]]] = {}
        for cid in cand_ids:
            cg = context.get_candidate_graph(cid)
            if not cg:
                continue
            for m in cg.node_mentions:
                order_key = (m.timestamp_seconds, m.utterance_index)
                cur = first_mention_owner.get(m.node_id)
                if cur is None or order_key < cur[1]:
                    first_mention_owner[m.node_id] = (cid, order_key)

        for cid in cand_ids:
            cg = context.get_candidate_graph(cid)
            if not cg:
                continue
            for m in cg.node_mentions:
                owner_info = first_mention_owner.get(m.node_id)
                if owner_info and owner_info[0] != cid and (m.timestamp_seconds, m.utterance_index) > owner_info[1]:
                    adoptions_map[owner_info[0]] += 1

        events = self._collect_chronological_events(context)
        m_count = len(events)
        if m_count < 2:
            return peer_alpha_map, self_alpha_map, adoptions_map

        spk_to_idx = {cid: i for i, cid in enumerate(cand_ids)}
        k_spk = len(cand_ids)
        ev_spk = np.array([spk_to_idx[u.speaker_id] for u in events], dtype=np.int32)

        # 构建以平均话轮间隔为单位的单调时间轴 τ（相邻话轮平均间隔 Δτ ≈ 1.0，兼容短讨论与45分钟长讨论）
        raw_t = np.array([float(u.start_seconds) for u in events], dtype=np.float64)
        for idx in range(1, m_count):
            if raw_t[idx] <= raw_t[idx - 1]:
                raw_t[idx] = raw_t[idx - 1] + 1.0
        mean_gap = max(1.0, (raw_t[-1] - raw_t[0]) / max(1, m_count - 1))
        tau = (raw_t - raw_t[0]) / mean_gap
        t_horizon = float(tau[-1]) + 1.0
        beta = 0.65  # 指数时间衰减率（覆盖后续 1~4 个话轮的激发窗口）

        # 计算耦合权重矩阵 W_sem (M x M): 基础话轮激发底数(对齐V5 diver) + 句向量余弦相似度 + 节点接力 + 显式互动回应
        emb = context.embedding_engine or get_embedding_engine()
        texts = [u.text[:400] for u in events]
        vecs = emb.embed_texts(texts) if emb is not None else np.ones((m_count, 1), dtype=np.float32)
        sim_mat = (0.15 + 0.85 * np.clip(np.dot(vecs, vecs.T), 0.0, 1.0)).astype(np.float64)

        # 叠加节点跨轮传递权重与显式互动边耦合权重
        turn_nodes: List[Set[str]] = [set() for _ in range(m_count)]
        u_idx_to_pos = {(u.speaker_id, u.index): pos for pos, u in enumerate(events)}
        for cid in cand_ids:
            cg = context.get_candidate_graph(cid)
            if cg:
                for m in cg.node_mentions:
                    pos = u_idx_to_pos.get((cid, m.utterance_index))
                    if pos is not None:
                        turn_nodes[pos].add(m.node_id)

        directed_reply_pairs = {(e.target, e.source) for e in context.interaction_network.edges}
        for i in range(m_count):
            for j in range(i + 1, m_count):
                if turn_nodes[i] and turn_nodes[i].intersection(turn_nodes[j]):
                    sim_mat[i, j] += 0.5
                if (events[i].speaker_id, events[j].speaker_id) in directed_reply_pairs:
                    sim_mat[i, j] += 0.6

        # 严格上三角时间衰减核 G[i, j] = I(i < j) * W_sem[i, j] * exp(-beta * (tau_j - tau_i))
        dt = tau[np.newaxis, :] - tau[:, np.newaxis]
        upper_mask = np.triu(np.ones((m_count, m_count), dtype=np.float64), k=1)
        G_kernel = upper_mask * sim_mat * np.exp(-beta * np.maximum(0.0, dt))

        # 积分分母项 ∫_{tau_i}^{T} exp(-beta*(t - tau_i)) dt = (1 - exp(-beta*(T - tau_i))) / beta
        integral_per_event = (1.0 - np.exp(-beta * (t_horizon - tau))) / beta  # (M,)
        spk_one_hot = np.eye(k_spk, dtype=np.float64)[ev_spk]  # (M, K)
        denom_per_sender = spk_one_hot.T @ integral_per_event + 1e-3  # (K,)

        # 初始化基线强度 mu (K,) 与激发矩阵 A[sender, receiver] (K, K)
        mu = np.full(k_spk, 0.25, dtype=np.float64)
        A = np.full((k_spk, k_spk), 0.20, dtype=np.float64)

        # 最多 15 步向量化 EM 迭代（带早停，纯内存 < 1ms），保证参数充分收敛
        for _ in range(15):
            # pair_alpha[i, j] = A[ev_spk[i], ev_spk[j]]
            pair_alpha = A[ev_spk[:, np.newaxis], ev_spk[np.newaxis, :]]
            contrib = pair_alpha * G_kernel  # (M, M), i -> j 对事件 j 的激发贡献
            lam = mu[ev_spk] + contrib.sum(axis=0)  # (M,)
            lam = np.maximum(lam, 1e-8)

            # 责任矩阵 P[i, j]: 事件 j 由历史事件 i 激发的后验概率
            P = contrib / lam[np.newaxis, :]
            p_bg = mu[ev_spk] / lam  # 背景自发概率

            # M-step 更新 mu 与激发矩阵 A
            mu = np.maximum(0.05, (spk_one_hot.T @ p_bg) / t_horizon)
            num_A = spk_one_hot.T @ P @ spk_one_hot  # (K, K)
            new_A = num_A / denom_per_sender[:, np.newaxis]
            if float(np.max(np.abs(new_A - A))) < 1e-4:
                A = new_A
                break
            A = new_A

        for idx, cid in enumerate(cand_ids):
            self_alpha_map[cid] = float(A[idx, idx])
            peer_sum = float(np.sum(A[idx, :]) - A[idx, idx])
            peer_alpha_map[cid] = peer_sum

        return peer_alpha_map, self_alpha_map, adoptions_map

    @staticmethod
    def _minmax_unit(values: Dict[str, float]) -> Dict[str, float]:
        """对齐 V5 s4_influence: 将自激/互激参数各自独立 MinMax 归一化到 [0.0, 1.0]"""
        if not values:
            return {}
        min_v = min(values.values())
        max_v = max(values.values())
        if max_v <= min_v:
            return {k: (0.5 if v > 0 else 0.0) for k, v in values.items()}
        span = max_v - min_v
        return {k: (v - min_v) / span for k, v in values.items()}

    def evaluate_all(self, context: ScoringContext) -> Dict[str, DimensionScore]:
        peer_alpha_map, self_alpha_map, adoptions_map = self._fit_multivariate_hawkes(context)
        norm_peer_map = self._minmax_unit(peer_alpha_map)
        norm_self_map = self._minmax_unit(self_alpha_map)
        candidate_ids = context.candidate_ids
        raw_results: Dict[str, tuple[float, Dict[str, Any], List[str]]] = {}

        for spk_id in candidate_ids:
            raw_results[spk_id] = self._score_speaker(
                spk_id, context, peer_alpha_map, self_alpha_map, norm_peer_map, norm_self_map, adoptions_map
            )

        all_raw_vals = [r[0] for r in raw_results.values()]
        out: Dict[str, DimensionScore] = {}
        for spk_id in candidate_ids:
            raw_val, metrics, key_points = raw_results[spk_id]
            out[spk_id] = DimensionScore(
                dimension_id=self.SCORER_ID,
                dimension_name=self.SCORER_NAME,
                score=normalize_to_standard_score(raw_val, all_raw_vals),
                raw_score=round(float(raw_val), 3),
                key_points=key_points,
                metrics=metrics,
            )
        return out

    def _score_speaker(
        self,
        speaker_id: str,
        context: ScoringContext,
        peer_alpha_map: Dict[str, float],
        self_alpha_map: Dict[str, float],
        norm_peer_map: Dict[str, float],
        norm_self_map: Dict[str, float],
        adoptions_map: Dict[str, int],
    ) -> tuple[float, Dict[str, Any], List[str]]:
        speaker = context.get_speaker(speaker_id)
        if not speaker or not speaker.utterances:
            return 0.0, {"hawkes_peer_excitation": 0.0, "hawkes_self_excitation": 0.0, "subsequent_node_adoptions": 0}, []

        peer_alpha = peer_alpha_map.get(speaker_id, 0.0)
        self_alpha = self_alpha_map.get(speaker_id, 0.0)
        norm_peer = norm_peer_map.get(speaker_id, 0.0)
        norm_self = norm_self_map.get(speaker_id, 0.0)
        node_adoptions = adoptions_map.get(speaker_id, 0)

        in_interactions = [e for e in context.interaction_network.edges if e.target == speaker_id]
        support_in = sum(1 for e in in_interactions if e.type == "support")
        total_in = len(in_interactions)

        # 严格对齐 V5 50%/50% 公式: MinMax(self_alpha) + MinMax(peer_alpha) ∈ [0.0, 2.0]（消除节点采纳/互动双重计数）
        total_raw = norm_self + norm_peer

        key_points = []
        if peer_alpha > 0.15:
            key_points.append(f"多变量 Hawkes 互激强度达 {peer_alpha:.2f}，显著带动后续成员围绕其议题展开推演")
        if node_adoptions > 0:
            key_points.append(f"率先抛出的核心论点被其他成员在后续讨论中跟进采纳 {node_adoptions} 次")
        if total_in > 0 and len(key_points) < 2:
            key_points.append(f"引发团队成员针对性互动回应 {total_in} 次")
        if not key_points and total_raw > 0:
            key_points.append("在群体讨论时序中产生了一定的话题启发效应")

        metrics = {
            "hawkes_peer_excitation": round(peer_alpha, 4),
            "hawkes_self_excitation": round(self_alpha, 4),
            "norm_peer_excitation": round(norm_peer, 4),
            "norm_self_excitation": round(norm_self, 4),
            "subsequent_node_adoptions": node_adoptions,
            "total_responses_received": total_in,
            "support_received": support_in,
        }
        return float(total_raw), metrics, key_points

    def calculate_raw_metrics(
        self,
        speaker_id: str,
        context: ScoringContext,
    ) -> tuple[float, Dict[str, Any], List[str]]:
        peer_map, self_map, adopt_map = self._fit_multivariate_hawkes(context)
        norm_peer_map = self._minmax_unit(peer_map)
        norm_self_map = self._minmax_unit(self_map)
        return self._score_speaker(speaker_id, context, peer_map, self_map, norm_peer_map, norm_self_map, adopt_map)
