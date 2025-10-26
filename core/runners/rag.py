# xpipe/runners/rag.py
# ======================================================================
# RAG runner with:
#   • pluggable retriever: simple_overlap | jaccard
#   • optional judge (enabled via llms.judge.enabled)
#   • prompt budgeting to avoid small-context model crashes
#   • logs per-stage tokens, latency, and quality metrics
# ======================================================================
