# Phase Checklists

Reference for deferred phases and future work.

---

## Phase 6: Security Hardening & Polish (IN PROGRESS)

### 6.1 Error Handling
- [x] Graceful degradation on API failures
- [x] Automatic retry with backoff (in BaseWorker)
- [x] Circuit breaker for external services (src/utils/circuit_breaker.py)
- [ ] Dead letter queue for failed tasks (deferred)

### 6.2 Monitoring
- [x] Prometheus metrics exporter (/metrics endpoint)
- [ ] Grafana dashboards (deferred - needs deployment)
- [ ] Alerting rules (deferred - needs deployment)
- [x] Cost tracking and budgets (in worker metrics)

### 6.3 Documentation
- [x] README.md with setup instructions
- [x] API documentation (in README)
- [x] Architecture decision records (docs/ARCHITECTURE.md)
- [x] Troubleshooting guide (in README)

### 6.4 Demo & Testing
- [x] One-command setup script (scripts/setup.sh)
- [x] Example research queries (scripts/demo.sh)
- [ ] Benchmark suite (deferred)
- [ ] Load testing (deferred)

### 6.5 Security (NEW)
- [x] Input validation hardening (src/security/validation.py)
- [x] Query sanitization and injection detection
- [x] URL safety validation
- [x] File path traversal protection
- [x] Test suite for security (tests/test_security.py)

---

## Future Phases (Deferred)

### Data Pipeline (for fine-tuning)
- ArXiv S3 bulk download
- LaTeX source extraction (TexSoup)
- Text cleaning, deduplication (MinHash LSH)
- JSONL formatting

### Continual Pre-Training (CPT)
- Physics/astro corpus (~250K papers)
- QLoRA: rank 64, alpha 128
- Unsloth for 2x speed
- Brev A100 (~12 hours)

### Supervised Fine-Tuning (SFT)
- Research methodology examples (~10K)
- Paper writing examples
- Search query examples
- Brev A100 (~4 hours)

### Self-Evolution System
- Trajectory logging format
- Success/failure labeling
- Critical step identification (ATLAS)
- Quality scoring
- Validation gates (factual, reasoning, diversity)
- 60/30/10 training corpus rule
- Staged deployment with rollback

---

## Completed Phase Details

See [PROGRESS.md](PROGRESS.md) for implementation notes on Phases 1-5.
