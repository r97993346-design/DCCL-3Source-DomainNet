"""Independent Optuna TPE with feasible categorical choices."""

import itertools

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import TrialState

from protocol import SAMPLER_SEED, SPACE, read_json, require, signature, write_json


class UniqueTPE(TPESampler):
    def __init__(self, rng_state):
        require(optuna.__version__ == "4.9.0", "the independent categorical mask was audited on Optuna 4.9.0")
        super().__init__(multivariate=False, n_startup_trials=5, n_ei_candidates=24, seed=SAMPLER_SEED)
        self.remaining = set(itertools.product(*SPACE.values()))
        self.rng_state = rng_state
        if rng_state.exists():
            self._restore_rng(read_json(rng_state))

    def reserve(self, params):
        self.remaining.discard(signature(params))

    def sample_independent(self, study, trial, param_name, distribution):
        keys = list(SPACE)
        prefix = {keys.index(name): value for name, value in trial.params.items()}
        index = keys.index(param_name)
        feasible = {combo[index] for combo in self.remaining
                    if all(combo[i] == value for i, value in prefix.items())}
        allowed = [value for value in distribution.choices if value in feasible]
        require(allowed, f"no feasible value for {param_name} and {trial.params}")
        if len(allowed) == len(distribution.choices):
            return super().sample_independent(study, trial, param_name, distribution)
        if len(allowed) == 1:
            return allowed[0]
        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED))
        if len(trials) < self._n_startup_trials:
            narrowed = optuna.distributions.CategoricalDistribution(allowed)
            return self._random_sampler.sample_independent(study, trial, param_name, narrowed)
        # Fit normal independent TPE on the full original domain. Restrict the
        # 24 candidates drawn from l(x) to values with an unseen continuation.
        below, above = _split_trials(study, trials, self._gamma(len(trials)), False)
        space = {param_name: distribution}
        good = self._build_parzen_estimator(study, space, below, handle_below=True)
        bad = self._build_parzen_estimator(study, space, above, handle_below=False)
        indices = np.array([distribution.to_internal_repr(v) for v in allowed])
        log_l = good.log_pdf({param_name: indices})
        probabilities = np.exp(log_l - np.max(log_l))
        candidates = self._rng.rng.choice(indices, size=self._n_ei_candidates,
                                          p=probabilities / probabilities.sum())
        samples = {param_name: candidates}
        selected = self._compare(samples, self._compute_acquisition_func(samples, good, bad))[param_name]
        return distribution.to_external_repr(selected)

    def checkpoint_rng(self):
        states = []
        for rng in (self._rng.rng, self._random_sampler._rng.rng):
            algorithm, keys, position, cached, gaussian = rng.get_state()
            states.append([algorithm, keys.tolist(), position, cached, gaussian])
        write_json(self.rng_state, states)

    def _restore_rng(self, states):
        for rng, state in zip((self._rng.rng, self._random_sampler._rng.rng), states):
            algorithm, keys, position, cached, gaussian = state
            rng.set_state((algorithm, np.array(keys, dtype=np.uint32), position, cached, gaussian))
