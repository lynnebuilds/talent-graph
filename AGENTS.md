# TalentGraph — AI Collaboration & Harness Guide (`AGENTS.md`)

> **Project**: TalentGraph — Pure-CPU (zero GPU/PyTorch) Leaderless Group Discussion (LGD) quantitative assessment engine based on Causal Influence Graphs, Interaction Topology, and 16-dimension pure-math scoring.
> **Principle**: Map, not encyclopedia. Follow these guardrails strictly; read deep docs on demand.

---

## 1. Repository Map & Dependency Direction

```text
talent-graph/
├── cli.py                       # Unified Typer CLI (download-model / prepare / evaluate / server / benchmark)
├── frontend/dist/               # Static web dashboard & print templates (index.html, report-print.html; no npm build)
├── data/                        # Runtime data root (cache/corpus, input, projects/{task_id}); decoupled from code
├── docs/                        # Deep architecture & contract docs (read on demand)
└── backend/
    ├── pyproject.toml           # Dependencies, pytest, ruff config (Python >= 3.10)
    ├── server.py                # FastAPI entrypoint (/health, /api/v1/*, /ws/assessment/*, static mount)
    ├── configs/settings.py      # Pydantic BaseSettings global config
    ├── src/
    │   ├── models/              # Pydantic V2 domain schemas (graph, discussion, score, report)
    │   ├── core/                # Infrastructure (llm_client, pure-CPU ONNX/API embedding, logger)
    │   ├── knowledge/           # Stage 0: StandardGraph extractor & SHA256 0s cache
    │   ├── pipeline/            # Stage 1-2: parser, interaction, node_matcher, arc_verifier, runner
    │   ├── scoring/             # Stage 3: 16-dim pure-math scorers (d1..d6) & ScoringEngine
    │   ├── reporting/           # Stage 4-5: single-pass summarizer, ReportGenerator, Playwright PDF
    │   └── api/                 # FastAPI v1 routes & WebSocket progress broadcaster
    └── tests/                   # Pytest unit & benchmark suite
```

**Unidirectional Dependency Rule** (upstream must NEVER import downstream):

```text
configs / models  ←  core  ←  knowledge / pipeline / scoring / reporting  ←  api / cli
```

---

## 2. Non-Negotiable Guardrails

1. **Pure-Math, In-Memory Scoring (`backend/src/scoring/`)**:
   - Zero LLM calls, zero disk I/O, zero matplotlib/image rendering inside scorers.
   - All 16 dimensions for 12 candidates must run in **< 50ms** (baseline ~2.5ms).
   - Normalized dimension scores stay in **`60.0 ~ 95.0`** (`normalize_to_standard_score`) unless explicitly changed by spec.
2. **Pure-CPU Architecture**:
   - Never introduce PyTorch, CUDA, Transformers, or ModelScope. Embeddings run via pure-CPU ONNX or cloud API (`EmbeddingEngine`).
3. **Unified LLM Governance**:
   - All LLM calls MUST go through `backend/src/core/llm_client.py` (`LLMClient` with `asyncio.Semaphore`, timeout, exponential backoff retry, and Pydantic structured parsing). Do not scatter raw OpenAI/HTTP clients in business modules.
4. **Frontend-Backend Contract Sync**:
   - `ReportData` (`backend/src/models/report.py`) is the single JSON contract consumed by REST API, `frontend/dist/index.html`, and `frontend/dist/report-print.html`.
   - Any field addition, rename, or removal in `ReportData` MUST update affected frontend HTML files and `docs/DATA_SCHEMA_AND_INTEGRATION.md`.
5. **Stateless Code / Isolated Data**:
   - All runtime artifacts (corpus cache, `report.json`, graphs, PDFs) MUST be written under `data/` via `Settings` path helpers—never inside `backend/src/`.
6. **Security & Side-Effect Discipline**:
   - Never hardcode API keys or credentials (read via `Settings` / `.env`).
   - Keep `mock_email=True` and `enable_telemetry=False` by default; no external tracking side effects.

---

## 3. Deterministic Verification Commands

Project Python environment lives in `backend/.venv`. Prefer the unified verification gate (`./scripts/check.sh`), or run individual tools from `backend/` (so `pyproject.toml` `pythonpath = ["."]` and `asyncio_mode = "auto"` apply):

```bash
# Unified gate (recommended)
./scripts/check.sh          # Full gate: ruff + mypy + pytest(--cov) + benchmark
./scripts/check.sh fast     # Fast TDD inner loop: ruff + pytest

# Individual commands
cd backend && .venv/bin/pytest tests/ -v
cd backend && .venv/bin/pytest tests/ --cov=src --cov-report=term-missing
cd backend && .venv/bin/ruff check src configs tests ../cli.py
cd backend && .venv/bin/mypy src configs
backend/.venv/bin/python cli.py benchmark

# Server health smoke check (when touching API / WebSocket / frontend)
backend/.venv/bin/python cli.py server --port 8000
curl -sf http://localhost:8000/health
```

> **Rule**: Never claim a task is complete based on reading code alone. Run the relevant verification commands above and report real command outputs.

---

## 4. Harness Development Workflow

For non-trivial features, algorithm updates, or refactors (skip full ceremony for 1-line typos/copy fixes):

1. **Clarify First**: Resolve ambiguity, edge cases, fallback behavior, and Out-of-Scope before coding (e.g., `grill-me` / `/grill-me`).
2. **Spec & Vertical Slices**: Break work into 2–6 small, independently testable vertical slices with concrete acceptance criteria (e.g., `write-a-prd` → `prd-to-plan`). Avoid horizontal slicing.
3. **TDD Loop**: For core domain logic (scoring formulas, pipeline pruning/parsing, fallback paths, state transitions), follow one **Red → Green → Refactor** cycle per slice (`tdd`). Do not write all implementation first and backfill tests later.
4. **Verify & Clean-Context Review**: Pass all deterministic checks (`pytest` + `ruff` + `mypy` + `benchmark`), then review `git diff` against both the spec and this `AGENTS.md` from a fresh perspective.
5. **Compound Learning**: Whenever an AI mistake or subtle regression occurs, add a regression test or a concise rule here so it cannot recur.

**Stop & Ask Human Before**:
- Changing 16-dimension scoring formulas or V5 alignment semantics
- Breaking public REST/WebSocket routes or `ReportData` JSON schema
- Adding/upgrading dependencies in `backend/pyproject.toml`
- Performing destructive file/data operations or touching `.env` / secrets

---

## 5. Ubiquitous Language (Domain Glossary)

| Term | Meaning |
| :--- | :--- |
| **StandardGraph** | Reference causal influence graph extracted from expert case corpus (`corpus.txt`), cached by SHA256 fingerprint (Stage 0). |
| **`c` / `d` / `u` Nodes** | `c` = Concept/Fact (客观事实/约束), `d` = Decision/Action (决策举措), `u` = Utility/Goal (效用目标). |
| **Arc** | Directed logical edge in a graph (`info`: `c→d`, `causal`: `d→u`, or `c→c` / state update). |
| **Dual-ended Pruning** | Stage 2 optimization: only verify a standard arc `A→B` via LLM if the candidate mentioned **both** `A` and `B` ($O(V)$ instead of $O(V^2)$). |
| **CandidateGraph** | Per-candidate subgraph containing `matched_nodes`, `node_mentions` (with timestamps/quotes), `verified_arcs`, and `novel_nodes` (valuable ideas outside standard corpus). |
| **InteractionNetwork** | Directed weighted interpersonal graph (`support` / `deny` / `neutral`) extracted across the transcript in Stage 1. |
| **ScoringContext / ScoreMatrix** | Read-only in-memory input and 16-dimension (`1_1`..`6_3`) normalized output of `ScoringEngine`. |
| **ReportData** | Self-contained JSON contract (`data/projects/{task_id}/report.json`) powering both web dashboard and Playwright A4 PDF export. |
| **V5** | Legacy GPU-bound assessment system used as algorithm accuracy baseline (e.g., Hawkes EM excitation in `4`, convergence delta in `6_3`). |

---

## 6. Deep Docs (Load On Demand)

- Multi-agent pipeline & sub-agent roles: `docs/AGENT_ARCHITECTURE.md`
- `ReportData` schema, TypeScript definitions & frontend integration: `docs/DATA_SCHEMA_AND_INTEGRATION.md`
- V5 legacy vs TalentGraph architecture & algorithm comparison: `docs/V5_VS_TALENT_GRAPH_COMPARISON.md`
- Dependencies & linter config: `backend/pyproject.toml`
- Environment variables template: `backend/.env.example`
