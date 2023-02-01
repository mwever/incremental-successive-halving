import math
import numpy as np
import random


class Candidate:
    def __init__(self, candidate):
        self.candidate = candidate
        self.performanceMap = dict()

    def storePerformance(self, budget, performance):
        self.performanceMap[budget] = performance

    def getPerformance(self, budget):
        return self.performanceMap[budget]

    def getCandidate(self):
        return self.candidate

    def __repr__(self):
        return str(self.candidate) + " Performances: " + str(self.performanceMap)


class BudgetTrackingPerformanceMeasure:
    def __init__(self, eval_func):
        self.eval_func = eval_func
        self.budget_acc = 0.
        self.invoc_acc = 0

    def evaluate(self, candidate, budget):
        self.invoc_acc += 1
        self.budget_acc += budget
        return self.eval_func(candidate, budget)

    def getAccumulatedBudget(self):
        return self.budget_acc

    def resetAccumulatedBudget(self):
        self.budget_acc = 0.


class SuccessiveHalving:

    def __init__(self, s, min_budget, max_budget, eta, eval_func, minimize=True, debug=True):
        self.s = s
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.eval_func = eval_func
        self.debug = debug
        self.minimize = minimize

        self.old_candidates = list()

    def successiveHalving(self, candidates):
        raise Exception("Not implemented yet!")

    def isBetter(self, challenger, incumbent, budget):
        if self.minimize:
            return challenger.getPerformance(budget) < incumbent.getPerformance(budget)
        else:
            return challenger.getPerformance(budget) > incumbent.getPerformance(budget)

    def increaseMaximumBudget(self):
        self.max_budget = self.max_budget * self.eta
        self.s = self.s + 1

    def getBestCandidateForMaximumBudget(self):
        best = None
        for c in self.old_candidates:
            if self.max_budget in c.performanceMap:
                if best is None or self.isBetter(challenger=c, incumbent=best, budget=self.max_budget):
                    best = c
        return best

    def computeParams(self, candidate_list, i):
        n_i = len(candidate_list)
        r_i = self.min_budget * (self.eta ** i)
        r_i1 = r_i * self.eta
        k = int(math.floor(n_i / self.eta))
        return n_i, r_i, r_i1, k

    def getFirstIterationN(self):
        return len(self.old_candidates)


class EfficientSuccessiveHalving(SuccessiveHalving):

    def __init__(self, s, min_budget, max_budget, eta, eval_func, minimize=True, debug=True):
        super().__init__(s, min_budget, max_budget, eta, eval_func, minimize, debug)

    def successiveHalving(self, candidates):
        candidate_list = list()
        for c in self.old_candidates:
            candidate_list.append(c)
        for c in candidates:
            candidate_list.append(c)
            self.old_candidates.append(c)

        for i in range(self.s+1):
            n_i, r_i, r_i1, k = self.computeParams(candidate_list, i)

            promotions = list()

            # collect all candidates that have already been evaluated for a higher budget
            for c in candidate_list:
                if r_i1 in c.performanceMap:
                    promotions.append(c)
            for c in promotions:
                candidate_list.remove(c)

            for c in candidate_list:
                if r_i not in c.performanceMap:
                    score = self.eval_func(candidate=c.getCandidate(), budget=r_i)
                    c.storePerformance(budget=r_i, performance=score)

            if self.debug:
                print("n_i", n_i, " candidates ", len(candidate_list))
            candidate_list = sorted(candidate_list, key=lambda c: c.getPerformance(r_i), reverse=self.minimize)
            for j in range(len(promotions), k):
                promotions.append(candidate_list.pop())

            candidate_list = promotions
        return self.getBestCandidateForMaximumBudget()


class ConservativeSuccessiveHalving(SuccessiveHalving):

    def __init__(self, s, min_budget, max_budget, eta, eval_func, minimize=True, strict=False, debug=True):
        super().__init__(s, min_budget, max_budget, eta, eval_func, minimize, debug)
        self.strict = strict

    def successiveHalving(self, candidates):
        candidate_list = list()
        for c in self.old_candidates:
            candidate_list.append(c)
        for c in candidates:
            candidate_list.append(c)
            self.old_candidates.append(c)

        for i in range(self.s+1):
            n_i, r_i, r_i1, k = self.computeParams(candidate_list, i)

            for c in candidate_list:
                if r_i not in c.performanceMap:
                    score = self.eval_func(candidate=c.getCandidate(), budget=r_i)
                    c.storePerformance(budget=r_i, performance=score)

            if self.debug:
                print("n_i", n_i, " candidates ", len(candidate_list))
            # if not strictly conservative mode, then reuse already evaluated candidates which are not part of the list anymore
            if not self.strict:
                for c in self.old_candidates:
                    if c not in candidate_list and r_i in c.performanceMap:
                        candidate_list.append(c)
                if self.debug:
                    print("extended candidate list has size ", len(candidate_list))

            candidate_list = sorted(candidate_list, key=lambda c: c.getPerformance(r_i), reverse=self.minimize)

            # top k
            promotions = list()
            for j in range(k):
                promotions.append(candidate_list.pop())

            # set candidate list for next iteration
            candidate_list = promotions
        return self.getBestCandidateForMaximumBudget()


class IDHyperband:
    def __init__(self, max_budget, eta, eval_func, conservative=False, strict=False, debug=False):
        self.max_budget = max_budget
        self.eta = eta
        self.eval_func = eval_func
        self.conservative = conservative
        self.strict = strict
        self.debug = debug

        self.s_max = math.floor(math.log(self.max_budget, self.eta))
        self.b = (self.s_max + 1) * self.max_budget

        self.brackets = list()
        for s in reversed(range(self.s_max + 1)):
            r = self.max_budget / self.eta ** s

            if self.conservative:
                sha = ConservativeSuccessiveHalving(s=s, min_budget=r, max_budget=max_budget, eta=eta, eval_func=eval_func, strict=strict, debug=debug)
            else:
                sha = EfficientSuccessiveHalving(s=s, min_budget=r, max_budget=max_budget, eta=eta, eval_func=eval_func, debug=debug)
            self.brackets.append(sha)

    def setDebug(self, debug):
        self.debug = debug
        for i in range(len(self.brackets)):
            self.brackets[i].debug = True

    def incrementMaxBudget(self):
        self.max_budget = self.eta * self.max_budget
        self.s_max += 1
        self.b = (self.s_max + 1) * self.max_budget
        for i in range(len(self.brackets)):
            self.brackets[i].increaseMaximumBudget()

        if self.conservative:
            sha = ConservativeSuccessiveHalving(s=0, min_budget=self.max_budget, max_budget=self.max_budget, eta=self.eta, eval_func=self.eval_func, strict=self.strict, debug=self.debug)
        else:
            sha = EfficientSuccessiveHalving(s=0, min_budget=self.max_budget, max_budget=self.max_budget, eta=self.eta, eval_func=self.eval_func, debug=self.debug)
        self.brackets.append(sha)

    def hyperband(self, candidate_sampling):
        incumbent = None
        for i in range(len(self.brackets)):
            bracket = self.brackets[i]
            br_ratio = self.b / self.max_budget
            etas_ratio = (self.eta ** bracket.s) / (bracket.s + 1)
            n = math.ceil(br_ratio * etas_ratio)
            bracket_n = bracket.getFirstIterationN()
            rest_n = n - bracket_n

            if self.debug:
                print("s", bracket.s, "r", bracket.min_budget, "R", self.max_budget, "eta", self.eta)
                print(
                    f"Sample {rest_n} new candidates because we need {n} candidates in the first iteration of this bracket and {bracket_n} are already assigned to the bracket")

            # sample rest_n candidates
            candidates = candidate_sampling.get(bracket=i, n=rest_n)

            if self.debug:
                print("Sampled ", len(candidates), " new candidates")
            result_of_bracket = bracket.successiveHalving(candidates)

            if self.debug:
                print("Result of bracket ", i, ": ", result_of_bracket)
            if incumbent is None or result_of_bracket.getPerformance(self.max_budget) < incumbent.getPerformance(
                    self.max_budget):
                if self.debug:
                    print("The result of bracket ", i, " yielded a new incumbent with performance ", result_of_bracket.getPerformance(self.max_budget))
                if incumbent is not None:
                    if self.debug:
                        print("Replacing the old candidate with performance", incumbent.getPerformance(self.max_budget))
                incumbent = result_of_bracket
            else:
                if self.debug:
                    print("The result of bracket ", i, " could not manage to provide a new incumbent")

        return incumbent

