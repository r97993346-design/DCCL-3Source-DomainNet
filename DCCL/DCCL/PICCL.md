# PICCL: Paired-Intervention Causal Connectivity Learning

PICCL is registered as an independent DomainBed algorithm. It inherits DCCL initialization assets, reuses the frozen `pre_featurizer` as the reference encoder, and inserts a low-rank causal mediator projection between pooled backbone features and the classifier/projector.

Training flow: `x, x_2 -> featurizer/pre_featurizer -> PIRE/CDRM -> ISSL -> CMP -> classifier/projector losses`.

Inference flow: `x -> featurizer -> CMP -> classifier`; the reference encoder and prototype updates are not used during prediction.
