# -*- coding: utf-8 -*-
"""Unit tests for Knowledge Graph Extraction and Cache Manager"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from configs.settings import Settings
from src.knowledge.cache import GraphCacheManager
from src.knowledge.extractor import GraphExtractionPayload, StandardGraphExtractor


@pytest.mark.asyncio
async def test_extractor_with_mock_llm():
    """测试 StandardGraphExtractor 提取结构化图谱并正确映射类型"""
    mock_llm = MagicMock()
    mock_payload = GraphExtractionPayload(
        title="新能源车出海案例",
        nodes=[
            {"id": "c1", "name": "欧洲关税政策", "type": "c", "description": "外部壁垒", "keywords": ["关税", "欧洲"]},
            {"id": "d1", "name": "建立本地KD组装厂", "type": "d", "description": "出海策略", "keywords": ["组装厂", "KD"]},
            {"id": "u1", "name": "降低综合出海成本", "type": "u", "description": "核心效益", "keywords": ["成本", "利润"]},
        ],
        arcs=[
            {"source": "c1", "target": "d1", "type": "info", "description": "政策倒逼建厂决策"},
            {"source": "d1", "target": "u1", "type": "causal", "description": "本地建厂规避关税降低成本"},
        ],
    )
    mock_llm.chat_json = AsyncMock(return_value=mock_payload)

    extractor = StandardGraphExtractor(llm_client=mock_llm)
    corpus = "某新能源汽车企业计划拓展欧洲市场，面临高额反补贴关税..."
    graph = await extractor.extract_graph(corpus, corpus_id="auto_overseas")

    assert graph.title == "新能源车出海案例"
    assert len(graph.nodes) == 3
    assert len(graph.arcs) == 2
    assert graph.get_node("d1").type == "d"
    assert graph.get_node("u1").type == "u"
    assert len(graph.get_nodes_by_type("c")) == 1


@pytest.mark.asyncio
async def test_cache_manager_0s_hit(tmp_path: Path):
    """测试 CacheManager 在首次保存后的 0 秒缓存命中与重放"""
    mock_llm = MagicMock()
    mock_payload = GraphExtractionPayload(
        title="缓存测试案例",
        nodes=[{"id": "c1", "name": "背景事实", "type": "c"}],
        arcs=[],
    )
    mock_llm.chat_json = AsyncMock(return_value=mock_payload)

    settings = Settings(corpus_cache_dir=tmp_path / "cache")
    extractor = StandardGraphExtractor(llm_client=mock_llm, settings=settings)
    cache_mgr = GraphCacheManager(cache_dir=tmp_path / "cache", extractor=extractor, settings=settings)

    corpus_text = "这是一篇用于测试 0 秒缓存命中的标准题库参考语料内容。"

    # 1. 首次调用：触发 LLM 提取并写入缓存
    g1 = await cache_mgr.get_or_extract(corpus_text, corpus_id="test_cache")
    assert len(g1.nodes) == 1
    assert mock_llm.chat_json.call_count == 1

    # 2. 第二次调用：直接命中磁盘缓存，不调用 LLM
    g2 = await cache_mgr.get_or_extract(corpus_text, corpus_id="test_cache")
    assert len(g2.nodes) == 1
    assert mock_llm.chat_json.call_count == 1  # 依然为 1，说明 0 秒从缓存返回！
    assert g2.title == "缓存测试案例"


@pytest.mark.asyncio
async def test_adaptive_chunking_for_long_corpus():
    """测试超长语料 (>20000字) 自动触发自适应分块抽取与图谱合并去重"""
    mock_llm = MagicMock()

    # 模拟分块 1 抽出的子图
    chunk1_payload = GraphExtractionPayload(
        title="超长宏观案例",
        nodes=[
            {"id": "c1", "name": "市场下行压力", "type": "c", "description": "大环境", "keywords": ["市场"]},
            {"id": "d1", "name": "收缩非核心业务", "type": "d", "description": "降本动作", "keywords": ["收缩"]},
        ],
        arcs=[{"source": "c1", "target": "d1", "type": "info"}],
    )
    # 模拟分块 2 抽出的子图 (包含重合决策与最终效用)
    chunk2_payload = GraphExtractionPayload(
        title="超长宏观案例",
        nodes=[
            {"id": "d1", "name": "收缩非核心业务", "type": "d", "description": "剥离亏损", "keywords": ["剥离"]},
            {"id": "u1", "name": "现金流转正", "type": "u", "description": "生存目标", "keywords": ["现金流"]},
        ],
        arcs=[{"source": "d1", "target": "u1", "type": "causal"}],
    )

    # 支持多个 chunk 并发调用
    mock_llm.chat_json = AsyncMock(side_effect=[chunk1_payload, chunk2_payload] * 10)

    settings = Settings(corpus_auto_chunk_threshold=2000, corpus_chunk_size=1500, corpus_chunk_overlap=200)
    extractor = StandardGraphExtractor(llm_client=mock_llm, settings=settings)

    # 构造超过阈值的超长文本 (如 6000+ 字)
    long_corpus = "这是一段关于企业组织重构与业务战略转型的深度案例研讨材料。\n\n" * 150
    assert len(long_corpus) > 2000

    graph = await extractor.extract_graph(long_corpus, corpus_id="long_corp_1")

    # 验证去重合并效果：同名 d "收缩非核心业务" 应被智能合并
    c_nodes = graph.get_nodes_by_type("c")
    d_nodes = graph.get_nodes_by_type("d")
    u_nodes = graph.get_nodes_by_type("u")

    assert len(c_nodes) >= 1
    assert len(d_nodes) >= 1
    assert len(u_nodes) >= 1
    assert len(graph.arcs) >= 2

    # 验证边连接正常且不包含自环
    for a in graph.arcs:
        assert a.source != a.target
        assert graph.get_node(a.source) is not None
        assert graph.get_node(a.target) is not None


@pytest.mark.asyncio
async def test_v5_hierarchy_and_cross_chunk_completion():
    """验证 V5 一二三级抽象层级、parent_concept 自动挂靠、根因边过滤与跨分块骨架补边"""
    from src.knowledge.extractor import CrossChunkArcItem, CrossChunkArcPayload

    mock_llm = MagicMock()
    c1_payload = GraphExtractionPayload(
        title="银行支行转型案例",
        nodes=[
            {"id": "c1", "name": "支行经营转型战略", "type": "c", "level": "一级", "sub_type": "known"},
            {"id": "c2", "name": "同业竞争加剧机制", "type": "c", "level": "二级", "sub_type": "known"},
            {"id": "c3", "name": "近三年网点业绩持续下滑", "type": "c", "level": "三级", "sub_type": "known", "is_root": True},
        ],
        arcs=[{"source": "c2", "target": "c3", "type": "causal"}],  # 指向 is_root=True 的因果边应被自动过滤！
    )
    c2_payload = GraphExtractionPayload(
        title="银行支行转型案例",
        nodes=[
            {"id": "d1", "name": "开展新业务专项培训", "type": "d", "level": "三级"},
            {"id": "u1", "name": "提升员工服务意识", "type": "u", "level": "二级", "sub_type": "instrumental"},
            {"id": "u2", "name": "实现网点利润正增长", "type": "u", "level": "一级", "sub_type": "final"},
        ],
        arcs=[
            {"source": "d1", "target": "u1", "type": "causal"},
            {"source": "u1", "target": "u2", "type": "causal"},
        ],
    )
    cross_payload = CrossChunkArcPayload(
        arcs=[CrossChunkArcItem(source="c3", target="d1", type="info")]
    )

    mock_llm.chat_json = AsyncMock(side_effect=[c1_payload, c2_payload, cross_payload])
    settings = Settings(corpus_auto_chunk_threshold=100, corpus_chunk_size=110, corpus_chunk_overlap=10)
    extractor = StandardGraphExtractor(llm_client=mock_llm, settings=settings)

    text = ("第一段：支行近三年面临同业竞争加剧，网点业绩持续下滑，亟需转型。\n\n"
            "第二段：应当立即开展新业务专项培训以提升员工服务意识，最终实现网点利润正增长。") * 2
    graph = await extractor.extract_graph(text, corpus_id="bank_case")

    assert graph.has_explicit_levels
    assert len(graph.final_target_ids) == 1
    assert len(graph.instrumental_target_ids) == 1
    # 三级节点自动挂靠到二级概念 parent_concept
    l3_nodes = [n for n in graph.nodes if n.level == "三级"]
    assert all(bool(n.parent_concept) for n in l3_nodes)
    # 指向 root 节点的非法因果边已被过滤，且跨分块 info 边已被补上
    root_ids = {n.id for n in graph.nodes if n.is_root}
    assert all(not (a.type == "causal" and a.target in root_ids) for a in graph.arcs)
    assert any(a.type == "info" for a in graph.arcs)


@pytest.mark.asyncio
async def test_cross_chunk_batches_interleave_c_and_d_and_root_restricted_to_c():
    """P0-8/P0-9 回归：>60 节点跨块补边批次必须同时包含 c 与 d；is_root 仅限 c 节点；显式 sub_type 合并生效"""
    from src.knowledge.extractor import CrossChunkArcItem, CrossChunkArcPayload, ExtractedNode

    # 构造 70 个三级节点（35 c + 35 d）+ 2 个 u，验证跨块补边批次不会把 c 和 d 隔离开
    c_nodes = [
        ExtractedNode(id=f"c{i}", name=f"背景约束条件编号{i}详述", type="c", level="三级", sub_type="known")
        for i in range(1, 36)
    ]
    d_nodes = [
        ExtractedNode(
            id=f"d{i}",
            name=f"执行管理方案编号{i}详述",
            type="d",
            level="三级",
            is_root=(i == 1),  # 误标 d1 为 root，不应被当作 root_ids
        )
        for i in range(1, 36)
    ]
    u_nodes = [
        ExtractedNode(id="u1", name="短期运营效率提升", type="u", level="二级", sub_type=""),
        ExtractedNode(id="u2", name="长期净利润增长目标", type="u", level="一级", sub_type="final"),
    ]
    # 第二个分块补充 u1 的显式 sub_type="instrumental"
    p1 = GraphExtractionPayload(title="大题本案例", nodes=c_nodes + [u_nodes[0]], arcs=[])
    p2 = GraphExtractionPayload(
        title="大题本案例",
        nodes=d_nodes + [
            ExtractedNode(id="u1_dup", name="短期运营效率提升", type="u", level="二级", sub_type="instrumental"),
            u_nodes[1],
        ],
        arcs=[{"source": "d2", "target": "d1", "type": "causal"}],  # 指向误标 root 的 d1 不应被过滤
    )

    seen_prompts: list[str] = []

    async def fake_chat_json(*, prompt: str, response_model, **kwargs):
        seen_prompts.append(prompt)
        return CrossChunkArcPayload(arcs=[CrossChunkArcItem(source="c1", target="d1", type="info")])

    mock_llm = MagicMock()
    mock_llm.chat_json = AsyncMock(side_effect=fake_chat_json)
    extractor = StandardGraphExtractor(llm_client=mock_llm)

    graph, chunks_map = extractor._merge_extracted_payloads([p1, p2], corpus_id="big", corpus_hash="h")
    # d 节点不允许成为 is_root，指向 d1 的因果边保留
    d1 = graph.get_node("d1")
    assert d1 is not None and not d1.is_root
    assert any(a.source == "d2" and a.target == "d1" and a.type == "causal" for a in graph.arcs)
    # 后分块显式 instrumental 应覆盖先分块的默认 final
    u1 = graph.get_node("u1")
    assert u1 is not None and u1.sub_type == "instrumental"

    graph = await extractor._complete_cross_chunk_arcs(graph, chunks_map)
    assert len(seen_prompts) >= 2
    # 每一个跨块补边批次中都应同时包含 c 和 d 节点
    for p in seen_prompts:
        assert "(c/" in p and "(d/" in p


