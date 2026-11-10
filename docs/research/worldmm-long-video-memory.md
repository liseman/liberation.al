# WorldMM long-video memory: historical research

WorldMM was read, not installed or run. Source: [WorldMM](https://github.com/wgcyeo/WorldMM) (Yeo, Kim, Yoon, and Hwang, CVPR 2026, [arXiv 2512.02425](https://arxiv.org/abs/2512.02425)), read at commit `133a55b` (2026-07-30). It bears on goal 2 in [ROADMAP.md](../ROADMAP.md) (answer questions about what happened) and the evidence model in [DOMAIN-MODEL.md](../DOMAIN-MODEL.md).

> Historical research from October 2026. The upstream compatibility, licensing,
> and benchmark observations below were recorded then and have not been
> revalidated. Current implementation details belong in
> [Architecture](../ARCHITECTURE.md), not this research note.

## Design decision

The research recommended porting ideas rather than adding WorldMM as a dependency:

1. Summarize evidence at several time scales and retrieve across them.
2. Let a bounded search loop gather evidence before answering.
3. Preserve the raw evidence behind summaries and facts about people.
4. Consider graph retrieval or video embeddings only when measurement shows a gap.

The application now stores journal entries in Continuum and searches through the
memory log and vector worker. The old SQLite porting plan was removed. See
[Measurement](../MEASUREMENT.md) for evaluating retrieval against reviewed labels.

## What WorldMM is

Research code (about 8,800 lines of Python, plus vendored copies of HippoRAG and VLM2Vec) for answering questions about week-long first-person video. The benchmark it targets, EgoLifeQA, is six people wearing camera glasses for a week, which is close to what the pack records.

It builds three memories offline:

| Memory | Built from | Retrieved by |
|---|---|---|
| Episodic | A first-person caption per 30 seconds, written by an LLM from the dense video captions and transcript lines in that window. An LLM then summarizes 3 minute, 10 minute, and 1 hour windows, each from the level below. OpenIE triples are extracted from each caption. | HippoRAG (a knowledge graph with personalized PageRank) per level, then one LLM call ranks the candidates from all levels together, preferring fine levels for specific events and coarse ones for habits. |
| Semantic | Generalized triples (relationships, roles, preferences, habits) extracted from the episodic triples, each with the episodic evidence that supports it. New triples are consolidated against existing ones above a cosine similarity of 0.6, and an LLM decides whether to merge or update. | Personalized PageRank over the triple graph. |
| Visual | A VLM2Vec-V2 embedding per clip (16 frames). | Text-to-clip similarity, or a time range, after which frames are decoded from the video file and passed to the answering model. |

To answer, a reasoning model loops up to five times. Each round it returns JSON: either "answer", or "search" with one memory type and a keyword-style query. Results are deduplicated across rounds. Then the answering model gets the question plus all retrieved text and frames.

Reported accuracy (percent):

| System | EgoLifeQA | Average over five benchmarks |
|---|---|---|
| GPT-5 alone | 48.6 | 61.1 |
| Qwen3-VL-8B alone | | 51.6 |
| WorldMM with GPT-5 | 65.6 | 69.5 |
| WorldMM with Qwen3-VL-8B | 56.4 | 59.9 |

Ablation (WorldMM with GPT-5, average): episodic only 64.9, visual only 44.9, episodic and semantic 66.8, episodic and visual 66.9, all three 69.5. With the 8B model, a single time scale instead of four costs 6.1 points, and plain embeddings instead of graphs cost 4.4. The paper reports latency only in a figure, and gives no call counts or cost.

## Compatibility concerns recorded in October 2026

- **The environments conflict.** The pack is on Python 3.14, transformers 5.x, and torch 2.11. WorldMM requires Python 3.12 or later and pins transformers below 5, torchvision below 0.24 (an older torch), flash-attn, apex, and decord. It cannot be an extra, only a separate venv and process.
- **CUDA only.** flash-attn, apex, the cu128 torch index, and decord have no Apple Silicon path, so this dependency set would not provide the required Apple Silicon path.
- **No Ollama backend.** The LLM wrapper picks a backend by finding "gpt" or "qwen3" in the model name. One backend calls the OpenAI Responses API (`responses.parse`), the other loads Qwen3-VL-8B in process through transformers. Reaching Ollama needs a new provider. Whether Ollama's Responses endpoint handles its structured-output calls was not checked.
- **Hosted models by default.** The memory build defaults to gpt-5-mini and answering to GPT-5, which would send everyone's conversations to OpenAI. The local 8B variant scores about 9 points lower.
- **Batch only.** Memory comes from offline scripts. Indexing up to a time re-feeds every caption before it to HippoRAG. Nothing appends as the day goes on.
- **Slow answers.** Up to five reasoning calls, an entity extraction call per graph search, and a final answer. Fine for the dashboard, far too slow for a spoken reply.
- **It breaks the domain model.** Outputs carry no sources or producer (rule 10). People are name strings inside triples (rule 3). There is no way to purge one person's data short of a rebuild (rule 12), and a relabel in the dashboard would also mean a rebuild.
- **Its own clock.** Times are EgoLife's "DAY1" plus an `HHMMSSff` integer, not epoch seconds (rule 5).
- **GPU memory (estimated, not measured).** Qwen3-VL-8B, Qwen3-Embedding-4B, and VLM2Vec-V2 together need most of one RTX 3090, beside the pack's own models.

The original review recorded Apache 2.0 for WorldMM and VLM2Vec, and MIT for HippoRAG. Recheck the exact versions and licenses before adopting any of them.

## Evidence and measurement requirements

A fluent summary can repeat an incorrect caption or transcript. Keep model output
identified as interpretation, retain its source references, and report missing
evidence. Names should resolve through current identity records, and deleting
source evidence must invalidate derived records.

A published benchmark does not establish performance on this application's
recordings. Compare retrieval variants using reviewed questions, expected answers,
and source times. Keep the model, context budget, retrieval settings, latency, and
label revision with each result. Local model contention and spoken-response latency
need separate measurements from answer accuracy.
