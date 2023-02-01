from yahpo_gym import benchmark_set, local_config
import numpy as np
from idhb import *
import os
from py_experimenter.result_processor import ResultProcessor
from py_experimenter.experimenter import PyExperimenter
import time

local_config.init_config()
local_config.set_data_path("yahpodata")


class YAHPOEvaluationFunction:
    def __init__(self, bench, metric):
        self.bench = bench
        self.metric = metric

    def evaluate(self, candidate, budget):
        xs = candidate.get_dictionary()
        # print("evaluate candidate ", xs, " for budget " , budget)
        xs["epoch"] = int(budget)
        return (-1) * self.bench.objective_function(xs)[0][self.metric]


class YAHPOCandidateSampler:
    def __init__(self, cs, seed):
        self.bracket_random_state = dict()
        self.global_random_state = seed
        self.cs = cs

    def get(self, bracket, n):
        # retrieve random state to set seed for sampling
        if bracket in self.bracket_random_state:
            random_state = self.bracket_random_state[bracket]
        else:
            np.random.seed(self.global_random_state)
            self.global_random_state = np.random.randint(0, 2 ** 16 - 1)
            random_state = np.random.randint(0, 2 ** 16 - 1)
            self.bracket_random_state[bracket] = random_state
        self.cs.seed(seed=random_state)
        # draw a list of n candidates
        candidates = list()

        if n > 1:
            for c in self.cs.sample_configuration(n):
                candidates.append(Candidate(candidate=c))
        else:
            candidates.append(Candidate(candidate=self.cs.sample_configuration(n)))
        return candidates


def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_fields: dict):
    seed = int(keyfields['seed'])
    benchmark = keyfields['benchmark']
    instance = keyfields['instance']
    algo = keyfields['algo']
    metric = keyfields['metric']
    eta = int(keyfields['eta'])
    initial_max_budget = int(keyfields['initial_max_budget'])

    bench = benchmark_set.BenchmarkSet(benchmark)
    bench.set_instance(instance)
    cs = bench.get_opt_space(drop_fidelity_params=True)
    eval = BudgetTrackingPerformanceMeasure(YAHPOEvaluationFunction(bench, metric).evaluate)

    sampler = YAHPOCandidateSampler(cs=cs, seed=seed)
    if algo == "cid-hb":
        hb = IDHyperband(max_budget=initial_max_budget, eta=eta, eval_func=eval, conservative=True, strict=False)
    elif algo == "dcid-hb":
        hb = IDHyperband(max_budget=initial_max_budget, eta=eta, eval_func=eval, conservative=True, strict=True)
    else:
        hb = IDHyperband(max_budget=initial_max_budget, eta=eta, eval_func=eval, conservative=False)
    hb.hyperband(sampler)

    if algo == "ih-hb":
        sampler = YAHPOCandidateSampler(cs=cs, seed=seed)
        hb = IDHyperband(max_budget=initial_max_budget * eta, eta=eta, eval_func=eval)
        res = hb.hyperband(sampler)
    elif algo == "eid-hb" or algo == "cid-hb" or algo == "dcid-hb":
        hb.incrementMaxBudget()
        res = hb.hyperband(sampler)
    else:
        print("Not supported algo!")

    results = {
        'final_incumbent': str(res.getCandidate()),
        'performance': (-1) * res.performanceMap[initial_max_budget * eta],
        'total_budget': eval.getAccumulatedBudget()
    }
    result_processor.process_results(results)


experimenter = PyExperimenter(experiment_configuration_file_path="config/yahpo.cfg", database_credential_file_path="config/database_credentials.cfg")
# experimenter.fill_table_from_config()
time.sleep(2.4)
experimenter.execute(experiment_function=run_experiment, max_experiments=-1, random_order=True)
