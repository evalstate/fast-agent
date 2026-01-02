---
type: MAKER
name: reliable_classifier
worker: classifier
k: 3
max_samples: 10
match_strategy: normalized
red_flag_max_length: 20
---

    MAKER: Massively decomposed Agentic processes with K-voting Error Reduction.
    Implements statistical error correction through voting consensus.
    Multiple samples are drawn and the first response to achieve a k-vote
    margin wins, ensuring high reliability even with cost-effective models.
