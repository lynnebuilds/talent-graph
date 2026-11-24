# -*- coding: utf-8 -*-
"""Unit tests for Discussion Parser and Interaction Extractor"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.interaction import ExtractedInteractionItem, InteractionExtractor, InteractionPayload
from src.pipeline.parser import DiscussionParser

SAMPLE_DISCUSSION_TEXT = """
说话人 1 00:00
各位候选人请注意，现在讨论正式开始，时间45分钟。

说话人 2 00:15
我认为我们应当先明确本次案例的核心目标，即如何在有限预算内完成市场扩张。

说话人 3 00:45
我非常赞同2号候选人的观点。在目标明确之后，我建议我们分成线上与线下两个实施路径。

说话人 4 01:20
我不同意3号候选人的分阶段建议，因为预算非常紧张，同时铺开两条线会导致资金断裂。

说话人 2 02:10
4号的风险提示很到位。那我们折中一下，第一季度先聚焦线上渠道，等现金流为正再拓展线下。
"""


def test_discussion_parser():
    parser = DiscussionParser(host_speaker_id="speaker_1")
    transcript = parser.parse_text(SAMPLE_DISCUSSION_TEXT, group_id="test_group_1")

    assert transcript.group_id == "test_group_1"
    assert len(transcript.utterances) == 5
    # 说话人1是主持人，候选人应为 speaker_2, speaker_3, speaker_4
    assert len(transcript.speakers) == 4
    assert len(transcript.candidate_speakers) == 3

    # 检查发言人2的统计
    spk2 = transcript.speakers["speaker_2"]
    assert spk2.turn_count == 2
    assert spk2.total_words > 0
    assert not spk2.is_host

    # 检查主持人
    spk1 = transcript.speakers["speaker_1"]
    assert spk1.is_host


def test_auto_host_preserves_real_candidate_1_and_inline_turns():
    """验证 auto 模式不会误删真实参评的 1 号候选人，且支持单行内联发言格式"""
    unmoderated_text = """
    说话人 1 00:05 我认为支行业绩下滑的核心原因是产品老化，建议立刻梳理高收益理财新业务并加强员工激励。
    说话人 2 00:45 我同意1号的判断，除此之外还要加强周边社区的网格化获客营销。
    """
    parser = DiscussionParser(host_speaker_id="auto")
    transcript = parser.parse_text(unmoderated_text, group_id="unmod_group")

    assert len(transcript.speakers) == 2
    assert len(transcript.candidate_speakers) == 2
    assert not transcript.speakers["speaker_1"].is_host
    assert "speaker_1" in [s.speaker_id for s in transcript.candidate_speakers]



@pytest.mark.asyncio
async def test_interaction_extractor_mock_and_heuristic():
    parser = DiscussionParser(host_speaker_id="speaker_1")
    transcript = parser.parse_text(SAMPLE_DISCUSSION_TEXT, group_id="test_group_1")

    mock_llm = MagicMock()
    mock_payload = InteractionPayload(
        interactions=[
            ExtractedInteractionItem(source="speaker_3", target="speaker_2", type="support", evidence="我非常赞同2号候选人的观点"),
            ExtractedInteractionItem(source="speaker_4", target="speaker_3", type="deny", evidence="我不同意3号候选人的分阶段建议"),
            ExtractedInteractionItem(source="speaker_2", target="speaker_4", type="support", evidence="4号的风险提示很到位"),
        ]
    )
    mock_llm.chat_json = AsyncMock(return_value=mock_payload)

    extractor = InteractionExtractor(llm_client=mock_llm)
    net = await extractor.extract_interactions(transcript)

    assert len(net.speakers) == 3
    assert len(net.edges) == 3
    assert transcript.interaction_network is not None

    G = net.to_networkx()
    assert G.has_edge("speaker_3", "speaker_2")
    assert G.has_edge("speaker_4", "speaker_3")


@pytest.mark.asyncio
async def test_interaction_extractor_heuristic_fallback():
    parser = DiscussionParser(host_speaker_id="speaker_1")
    transcript = parser.parse_text(SAMPLE_DISCUSSION_TEXT, group_id="test_group_1")

    # 当 LLM 报错时自动触发启发式降级
    failing_llm = MagicMock()
    failing_llm.chat_json = AsyncMock(side_effect=RuntimeError("API error"))

    extractor = InteractionExtractor(llm_client=failing_llm)
    net = await extractor.extract_interactions(transcript)

    assert len(net.speakers) == 3
    # 启发式规则应识别出“赞同”和“不同意”
    assert len(net.edges) >= 2
    types = [e.type for e in net.edges]
    assert "support" in types
    assert "deny" in types


def test_parser_does_not_create_fake_speakers_and_parses_timestamp_colon_inline():
    """P0-12 回归：短冒号小标题不建假说话人；支持同时带时间戳与冒号的行内格式及 '1号候选人' 头行"""
    messy_text = """
    说话人 1 00:15：我认为要从三个方面展开。
    第一点：
    降低获客成本。
    总结如下：
    先做试点再推广。
    2号候选人
    [00:45] 说话人 2 我同意1号的试点思路。
    """
    parser = DiscussionParser(host_speaker_id="auto")
    transcript = parser.parse_text(messy_text, group_id="messy_group")

    assert set(transcript.speakers.keys()) == {"speaker_1", "speaker_2"}
    assert "第一点" in transcript.speakers["speaker_1"].full_text()
    assert "总结如下" in transcript.speakers["speaker_1"].full_text()
    assert "试点思路" in transcript.speakers["speaker_2"].full_text()


@pytest.mark.asyncio
async def test_interaction_normalizes_aliases_unknown_type_and_overlap_dedup():
    """P0-11 回归：非标类型不强转 support、权重钳位、中文别名映射、跨窗口同轮次去重"""
    parser = DiscussionParser(host_speaker_id="speaker_1")
    transcript = parser.parse_text(SAMPLE_DISCUSSION_TEXT, group_id="test_group_alias")

    mock_llm = MagicMock()
    mock_payload = InteractionPayload(
        interactions=[
            # 中文别名 + 非标反对词 + 超界权重
            ExtractedInteractionItem(
                source="候选人 4",
                target="3号候选人",
                type="oppose",
                utterance_index=3,
                evidence="我不同意3号",
                weight=9.9,
            ),
            # 未知类型应归 neutral 而非 support
            ExtractedInteractionItem(
                source="说话人 3",
                target="2",
                type="unknown_tag",
                utterance_index=2,
                evidence="顺便补充一句",
                weight=0.1,
            ),
            # 同一轮次 (src, tgt, type, u_idx) 不同 evidence 措辞应去重
            ExtractedInteractionItem(
                source="speaker_4",
                target="speaker_3",
                type="deny",
                utterance_index=3,
                evidence="另一窗口截出的不同引文",
                weight=1.5,
            ),
        ]
    )
    mock_llm.chat_json = AsyncMock(return_value=mock_payload)

    extractor = InteractionExtractor(llm_client=mock_llm)
    net = await extractor.extract_interactions(transcript)

    assert len(net.edges) == 2
    by_pair = {(e.source, e.target): e for e in net.edges}
    assert by_pair[("speaker_4", "speaker_3")].type == "deny"
    assert 0.5 <= by_pair[("speaker_4", "speaker_3")].weight <= 2.0
    assert by_pair[("speaker_3", "speaker_2")].type == "neutral"
    assert 0.5 <= by_pair[("speaker_3", "speaker_2")].weight <= 2.0

