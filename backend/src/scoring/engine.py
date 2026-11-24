# -*- coding: utf-8 -*-
"""
16-Dimension Quant Scoring Engine.
Orchestrates all 16 pure-mathematical & graph-theoretical scorers in memory.
Ultra-fast execution (<10ms per candidate, <50ms for the entire team).
"""

import time
from typing import Dict, List, Optional

from configs.settings import Settings, get_settings
from src.core.logger import get_logger
from src.models.score import ALL_DIMENSIONS, DimensionScore, ParticipantScoreProfile, ScoreMatrix
from src.scoring.base import BaseDimensionScorer, ScoringContext
from src.scoring.dimensions.d1_thinking import AbstractionScorer, LogicalReasoningScorer, SystematicThinkingScorer
from src.scoring.dimensions.d2_execution import GoalDecompositionScorer, ImpactPredictionScorer, InfoCollectionScorer
from src.scoring.dimensions.d3_creativity import AssociationScorer, CreativityScorer, FeasibilityScorer
from src.scoring.dimensions.d4_influence import InfluenceScorer
from src.scoring.dimensions.d5_teamwork import CollaborationCentralityScorer, ConflictResolutionScorer, OpennessScorer
from src.scoring.dimensions.d6_efficiency import ConvergenceScorer, FocusScorer, TimingControlScorer

logger = get_logger("scoring_engine")


class ScoringEngine:
    """16 维纯量化图论评分引擎"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.scorers: List[BaseDimensionScorer] = [
            # 维度一：思维能力
            LogicalReasoningScorer(),
            AbstractionScorer(),
            SystematicThinkingScorer(),
            # 维度二：决策能力
            GoalDecompositionScorer(),
            InfoCollectionScorer(),
            ImpactPredictionScorer(),
            # 维度三：创新能力
            CreativityScorer(),
            AssociationScorer(),
            FeasibilityScorer(),
            # 维度四：思维激发
            InfluenceScorer(),
            # 维度五：冲突协调
            OpennessScorer(),
            ConflictResolutionScorer(),
            CollaborationCentralityScorer(),
            # 维度六：推进能力
            FocusScorer(),
            TimingControlScorer(),
            ConvergenceScorer(),
        ]
        self.scorer_map = {s.SCORER_ID: s for s in self.scorers}

    @staticmethod
    def compute_total_score(dimension_scores: Dict[str, DimensionScore]) -> float:
        """
        对齐 V5 summary.py 总分计算方式：先按 6 大能力模块求类内均分，再取 6 大类等权均值
        （维度 4 占 1/6 ≈ 16.67%，其余 15 个子维度各占 1/18 ≈ 5.56%）。
        """
        if not dimension_scores:
            return 0.0
        cat_buckets: Dict[str, List[float]] = {}
        for dim_def in ALL_DIMENSIONS:
            if dim_def.id in dimension_scores:
                cat_buckets.setdefault(dim_def.category, []).append(dimension_scores[dim_def.id].score)
        if not cat_buckets:
            scores_list = [ds.score for ds in dimension_scores.values()]
            return round(sum(scores_list) / len(scores_list), 1)
        cat_means = [sum(vals) / len(vals) for vals in cat_buckets.values() if vals]
        return round(sum(cat_means) / len(cat_means), 1)

    def evaluate_all(self, context: ScoringContext) -> ScoreMatrix:
        """
        在纯内存中执行全量 16 个评分器，生成标准评分矩阵与团队大盘
        """
        start_time = time.perf_counter()
        logger.info(f"📊 [SCORING ENGINE] Starting 16-dimension evaluation for {len(context.candidate_ids)} candidates")

        candidate_ids = context.candidate_ids
        matrix = ScoreMatrix(group_id=context.group_id)

        # 初始化候选人画像
        for spk_id in candidate_ids:
            spk = context.get_speaker(spk_id)
            display_name = spk.display_name if spk else spk_id
            matrix.participants[spk_id] = ParticipantScoreProfile(
                speaker_id=spk_id,
                display_name=display_name,
                dimension_scores={},
            )

        # 逐个维度运行打分器 (每个维度内部对全场候选人计算并归一化)
        for scorer in self.scorers:
            dim_scores: Dict[str, DimensionScore] = scorer.evaluate_all(context)
            for spk_id, d_score in dim_scores.items():
                if spk_id in matrix.participants:
                    matrix.participants[spk_id].dimension_scores[scorer.SCORER_ID] = d_score

        # 计算每个候选人的综合加权总分（对齐 V5 6 大类等权均值）
        for spk_id, profile in matrix.participants.items():
            if profile.dimension_scores:
                profile.total_score = self.compute_total_score(profile.dimension_scores)

        # 聚合计算全场均分、极值与排名
        matrix.compute_aggregations()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"✅ [SCORING ENGINE COMPLETE] 16 dimensions calculated in {elapsed_ms:.2f}ms in-memory!")

        return matrix
