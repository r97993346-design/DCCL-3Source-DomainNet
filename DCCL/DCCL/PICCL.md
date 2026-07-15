# PICCL: Paired-Intervention Causal Connectivity Learning

PICCL is registered as an independent DomainBed algorithm. It inherits DCCL initialization assets, reuses the frozen `pre_featurizer` as the reference encoder, and inserts a causal feature module between pooled backbone features and the original DCCL classifier/projector losses.

The active causal feature path is:

```text
backbone feature z -> causal mediator -> residual gate fusion(z, mediator(z)) -> original DCCL losses
```

The DCCL loss semantics are intentionally preserved after fusion: CE, optional two-view CE, supervised contrastive loss, pre-trained feature contrastive loss, and the `l_d` pre-training anchoring regularizer are still composed by the shared `DCCL.update` loss flow. PICCL only overrides feature/extra-loss hooks, so the borrowed DCCL portion stays tied to the original implementation. PICCL adds only two causal auxiliary losses on top of those DCCL terms: `loss_isr` and `loss_orth`, weighted independently by `piccl_isr_weight` and `piccl_orth_weight`.

Training flow: `x, x_2 -> featurizer/pre_featurizer -> PIRE/CDRM responses -> ISR/orth losses -> mediator -> residual gate -> DCCL classifier/projector losses`.

Inference flow: `x -> featurizer -> mediator -> residual gate -> classifier`; the reference encoder and prototype updates are not used during prediction.
