"""B5c stochastic diversity-aware intervention prompt sampling for CIPT.

This module keeps the official-normalize-identity CIPTDCCL implementation
unchanged and only replaces how B5c intervention prompts are selected.

For every training *and* evaluation forward pass:
1) choose one B5c prompt uniformly at random;
2) repeatedly sample the next prompt with probability biased toward prompts
   that are far from the already-selected set in frozen CLIP text space;
3) stop after K distinct prompts have been selected.

The rule intentionally does not branch on ``module.training``. Therefore train
and test use the same stochastic intervention distribution. A fixed torch seed
still makes an experiment reproducible.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_dccl_ablation import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_prompt import CIPTTextFeatures


# Preserve the original selector once so B5a/B5b behavior remains untouched.
if not hasattr(CIPTTextFeatures, "_original_intervention_features"):
    CIPTTextFeatures._original_intervention_features = (
        CIPTTextFeatures.intervention_features
    )


def _stochastic_diverse_b5c_indices(text_features, bank):
    """Sample K distinct B5c prompts with stochastic max-min diversity bias.

    The first prompt is uniform random. For each subsequent draw, every
    remaining candidate receives a score equal to its minimum cosine distance
    to the already-selected prompts. Scores are converted into probabilities
    with a softmax temperature and one candidate is sampled with
    ``torch.multinomial``.

    Lower temperature -> stronger diversity preference / less randomness.
    Higher temperature -> weaker diversity preference / closer to uniform.
    """
    num_available = int(bank.shape[0])
    k = int(text_features.k)
    temperature = float(
        getattr(text_features, "stochastic_diversity_temperature", 0.1)
    )

    if num_available < 1:
        raise ValueError("B5c template bank must contain at least one prompt.")
    if k < 1:
        raise ValueError("cipt_k must be at least 1.")
    if k > num_available:
        raise ValueError(
            "Stochastic diversity sampling requires cipt_k <= B5c bank size; "
            "got K={} and bank size={}.".format(k, num_available)
        )
    if temperature <= 0.0:
        raise ValueError(
            "cipt_prompt_temperature must be positive; got {}.".format(
                temperature
            )
        )

    features = F.normalize(bank.float(), dim=-1)

    # Randomness is explicit from the first intervention onward.
    first = int(
        torch.randint(
            low=0,
            high=num_available,
            size=(1,),
            device=bank.device,
        ).item()
    )
    selected = [first]

    while len(selected) < k:
        selected_features = features[selected]
        similarities = features @ selected_features.transpose(0, 1)

        # max-min diversity: score each candidate by the distance to its
        # nearest already-selected prompt.
        min_distance = (1.0 - similarities).min(dim=1).values

        # Never sample a prompt twice in the same intervention set.
        min_distance[selected] = -torch.inf

        # Stochastic rather than argmax selection. Prompts farther from the
        # selected set receive larger probability but are not forced.
        logits = min_distance / temperature
        probabilities = torch.softmax(logits, dim=0)
        next_idx = int(
            torch.multinomial(probabilities, num_samples=1).item()
        )
        selected.append(next_idx)

    return torch.tensor(
        selected,
        dtype=torch.long,
        device=bank.device,
    )


def _stochastic_intervention_features(self, labels=None):
    """Use stochastic diversity sampling for B5c in both train and eval."""
    if self.template_mode != "b5c":
        return self._original_intervention_features(labels=labels)

    bank = self.b5c_text_bank
    indices = _stochastic_diverse_b5c_indices(self, bank)
    return bank.index_select(0, indices)


# Patch only the B5c branch of CIPTTextFeatures. The original method is still
# called for B5a and B5b, so existing ablation behavior remains available.
CIPTTextFeatures.intervention_features = _stochastic_intervention_features


class CIPTDCCL(_BaseCIPTDCCL):
    """Official-normalize-identity CIPTDCCL + stochastic diverse B5c prompts."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # This branch is specifically defined around the expanded B5c nuisance
        # bank. config.yaml also sets this value so logged hparams match the
        # actual execution path.
        hparams["cipt_template_mode"] = "b5c"

        super().__init__(input_shape, num_classes, num_domains, hparams)

        temperature = float(hparams.get("cipt_prompt_temperature", 0.1))
        if temperature <= 0.0:
            raise ValueError(
                "cipt_prompt_temperature must be positive; got {}.".format(
                    temperature
                )
            )

        self.text_features.stochastic_diversity_temperature = temperature

        print(
            "CIPT B5c stochastic-diversity prompts: bank_size={}, K={}, "
            "temperature={}, train_eval_same_sampling=True".format(
                int(self.text_features.b5c_text_bank.shape[0]),
                int(self.text_features.k),
                temperature,
            )
        )
