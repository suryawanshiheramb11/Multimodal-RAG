"""Multi-source timeline synchronization.

Aligns recordings with unknown time offsets by:
  - Audio fingerprinting: matching high-similarity AST embeddings (>0.9 cosine)
  - Visual matching: CLIP image embeddings with temporal consistency
  - Identity co-occurrence: same face cluster appearing in both sources

Outputs a unified case_time across all evidence nodes based on the reference source.
"""
