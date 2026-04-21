"""
benchmark_core.py
============
Benchmark optimization algorithms on MSG landscapes.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import csv
import torch
import numpy as np
from typing import Dict, List, Tuple, cast
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from pymoo.algorithms.soo.nonconvex.de   import DE
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination.default import DefaultSingleObjectiveTermination

from msg_utils import msg_eval
from ns_utils import decode_genome

# ─────────────────────────────────────────────────────────────────────────────
# pymoo problem
# ─────────────────────────────────────────────────────────────────────────────

class MSGProblem(Problem):
    def __init__(self, genome: torch.Tensor, means: torch.Tensor, device: str):
        self.num_gauss, self.dim = means.shape
        super().__init__(n_var=self.dim, xl=0.0, xu=1.0)
        self.means = means
        self.alphas, self.sigma = decode_genome(genome, self.num_gauss)
        self.device = device

    def _evaluate(self, X, out):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            f_X_t = msg_eval(X=X_t, means=self.means, alphas=self.alphas, sigma=self.sigma)
        if f_X_t.dim() == 1:
            f_X_t = f_X_t.unsqueeze(1)
        out["F"] = -1 * f_X_t.cpu().numpy()

def extract_history(res) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_X_list, best_F_list, n_eval_list = [], [], []
    for entry in res.history:
        F = entry.pop.get("F")   # (pop, 1)
        X = entry.pop.get("X")   # (pop, D)
        idx = np.argmin(F)
        best_F_list.append(float(F[idx]))
        best_X_list.append(X[idx])
        n_eval_list.append(entry.evaluator.n_eval)
    return np.array(best_X_list), np.array(best_F_list), np.array(n_eval_list)

# ─────────────────────────────────────────────────────────────────────────────
# benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _bench_one_trial(
    problem: MSGProblem,
    alg_name: str,
    global_opt: np.ndarray,
    seed: int,
    max_evals: int,
    coord_tol: float,
) -> Dict:
    rng = np.random.default_rng(seed)
    dim = len(global_opt)
    pop_size = max(10*dim, 20)

    if alg_name == "DE":
        alg = DE(pop_size=pop_size, variant="DE/rand/1/bin", CR=0.9, F=0.5, dither="vector", jitter=False)
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-11, ftol=1e-11, n_max_evals=max_evals
        )
    elif alg_name == "CMA-ES":
        alg = CMAES(x0 = rng.uniform(0, 1, dim), eval_initial_x = True, eval_final_mean= True)
        termination = ("n_eval", max_evals)

    res = minimize(problem, alg, termination=termination, seed=seed, verbose=False, save_history=True)
    best_X_hist, best_F_hist, n_eval_hist = extract_history(res)
    conv_eval = n_eval_hist[-1]
    successed = np.where(np.abs(best_X_hist - global_opt).max(axis=1)  < coord_tol)[0]
    succ_eval = int(n_eval_hist[successed[0]]) if len(successed) > 0 else None

    return {"conv_eval":conv_eval, "succ_eval": succ_eval}

def benchmark_one_problem(
    genome: torch.Tensor,
    means: torch.Tensor,
    device: str,
    n_trials: int,
    max_evals: int,
    coord_tol: float,
    seed:  int,
    alg_names:  List[str],
) -> Dict[str, Dict]:
    problem   = MSGProblem(genome=genome, means=means, device=device)
    alphas, _ = decode_genome(genome, len(means))
    global_opt = means[np.argmax(alphas)].squeeze(0).cpu().numpy()
    results   = {}
    for alg_name in alg_names:
        trial_results = []
        for t in range(n_trials):
            r = _bench_one_trial(problem=problem, alg_name=alg_name, global_opt=global_opt, seed=seed+t, 
                                 max_evals = max_evals, coord_tol = coord_tol)
            trial_results.append(r)
        conv_evals = [r["conv_eval"] for r in trial_results]
        succ_evals = [r["succ_eval"] for r in trial_results]
        n_success = sum(h is not None for h in succ_evals)

        success_rate = n_success / n_trials
        conv_time = sum(conv_evals) / n_trials

        results[alg_name] = {
            "success_rate": success_rate,
            "conv_time": conv_time,
        }

    return results

def run_benchmark(
    n_trials: int,
    max_evals: int,
    coord_tol: float,
    seed: int,
    bench_path: str,
    ns_path: str,
    alg_names: List[str],
    device: str
) -> None:
    os.makedirs(os.path.dirname(bench_path) if os.path.dirname(bench_path) else ".", exist_ok=True)
    data = torch.load(ns_path, weights_only=False)
    means = data["means"].to(device)
    all_genomes = data["_records_genomes"]
    if "all" not in bench_path:
        grid_cell_idx = data["grid_cell_idx_novelty"]
        cell_key = list(grid_cell_idx.keys())
        genomes = torch.stack([all_genomes[idx] for idx in grid_cell_idx.values()]).to(device)
    else:
        novelty_idx = [i for i, m in enumerate(data["_records_meta"]) if m["population"]=="novelty"]
        genomes = data["_records_genomes"][novelty_idx]
        features = data["_records_features"][novelty_idx]
        idxs = np.minimum(((features.numpy() / (1 + 1e-12)) * data["bins"] + 1e-12).astype(int), data["bins"] - 1)
        cell_key = [tuple(i) for i in idxs]
    n_problems = len(genomes)
    print(f"load from nsgrid (novelty): {n_problems} problems")

    # ── benchmarking ──────────────────────────────────────────────────
    metric_names = ["success_rate", "conv_time"]
    header = (["problem_id", "cell_key"]+ [f"{a}_{m}" for a in alg_names for m in metric_names])
    with open(bench_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    print("benchmarking...")
    if device == "cpu":
        n_jobs = max(1, os.cpu_count() // 2)
        with tqdm_joblib(total=n_problems) as _:
            bench = cast(list[dict], Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(benchmark_one_problem)(
                    genome=genomes[i], means=means, device=device,
                    n_trials=n_trials, max_evals=max_evals, coord_tol=coord_tol,
                    seed=seed, alg_names=alg_names) for i in range(n_problems)))
    else:
        bench = [
            benchmark_one_problem(
                genome=genomes[i], means=means, device=device,
                n_trials=n_trials, max_evals=max_evals, coord_tol=coord_tol,
                seed=seed, alg_names=alg_names)
            for i in range(n_problems)
        ]

    for i in range(n_problems):
        row   = ([i, cell_key[i]] + [bench[i][a][m] for a in alg_names for m in metric_names])
        with open(bench_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    print(f"\nend！ CSV saved: {bench_path}")