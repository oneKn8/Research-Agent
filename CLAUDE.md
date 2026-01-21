# STRICT MODE - Deep Research AI

**Status**: Phase 6 (Security Hardening) | [docs/PROGRESS.md](docs/PROGRESS.md)

---

## MANDATORY (violate = STOP immediately)

1. **NEVER GUESS** - Research first. Verify versions, syntax, configs with official docs.
2. **NEVER SKIP SECURITY** - Sanitize inputs, no hardcoded secrets, prevent injection.
3. **NEVER MENTION AI** - No Claude/Anthropic in commits, code, or docs.
4. **ALWAYS ASK** - When uncertain about requirements, ask. Don't assume.
5. **ALWAYS READ FIRST** - Understand existing code before modifying.

---

## Project Stack

- Brain: DeepSeek-R1-Distill-Qwen-14B (128K, vLLM, NO system prompts)
- Workers: GPT-4o-mini API
- Orchestration: LangGraph
- Output: LaTeX + BibTeX
- UI: Next.js 15 + Aceternity UI + Motion (BANNED: Streamlit, Gradio, shadcn)

---

## Code Standards

- No emojis anywhere
- No placeholder code
- Type hints, error handling, async timeouts
- Follow existing patterns
- Quality over speed

---

## Docs

| File | Content |
|------|---------|
| [docs/PROGRESS.md](docs/PROGRESS.md) | Status, notes |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design, stack |
| [docs/PHASE_CHECKLISTS.md](docs/PHASE_CHECKLISTS.md) | Phase 6 tasks |
| [BUILDPLAN.md](BUILDPLAN.md) | Implementation plan |
