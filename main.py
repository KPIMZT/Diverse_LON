import torch
from ns_core import NSGeneration
from ns_io import save_ns
from benchmark_core import run_benchmark
from cor_reg_core import run_correlation

if __name__ == "__main__":
    for dim in [2,5,10]:
        for seed in range(10):
            for init_parents in ["archetype", "random"]:
                seed = seed
                dim = dim
                num_gauss = 50*dim
                num_sample = 500 * dim
                r = num_gauss**(-1/dim)
                _mu = 20
                _lambda = 100
                generations = 100
                feature_type=["num_nodes", "global_funnel_size"]
                device="cuda" if torch.cuda.is_available() else "cpu"
                novelty_threshold=0.05
                novelty_k=15
                alpha_std=0.1
                alpha_min=0
                sigma_std=0.05
                sigma_max=3 * r
                sigma_min=r / 4
                use_crossover=False
                init_parents=init_parents
                plus_comma="plus"
                compute_random_baseline=True
                if init_parents == "archetype":
                    ns_path = f"./results_NS_trace/dim{dim}_seed{seed}.pt"
                else:
                    ns_path = f"./results_NS_trace/dim{dim}_seed{seed}_nonarch.pt"

                # Benchmark
                n_trials = 31
                max_evals = 1000*dim
                coord_tol = 1e-2
                seed = seed
                alg_names = ["CMA-ES, DE"]
                

                # Corr & Regg
                features = ['num_nodes', "edge_density", "num_sink", "avg_path_sinks", "avg_path_opt", "in_strength_sinks", "in_strength_opt", "global_funnel_size"]
                metrics = ["success_rate", "conv_time"]
                n_estimators= 200
                cv = 10

                ns = NSGeneration(num_gauss=num_gauss, dim=dim, _mu=_mu, _lambda=_lambda, generations=generations,
                                r=r, num_samples=num_sample, feature_type=feature_type, device=device, seed=seed,
                                novelty_threshold=novelty_threshold, novelty_k=novelty_k,
                                alpha_std=alpha_std, alpha_min=alpha_min, sigma_std=sigma_std, sigma_max=sigma_max, sigma_min=sigma_min,
                                use_crossover=use_crossover, init_parents=init_parents, plus_comma=plus_comma, compute_random_baseline=compute_random_baseline)
                ns.run()       
                ns.compute_coverage_all(scale=None, bins=30)
                save_ns(ns, ns_path)

            for alg in alg_names:
                bench_path = f"./results_Benchmark_trace/dim{dim}_seed{seed}_all_{alg}.csv"
                run_benchmark(n_trials=n_trials, max_evals=max_evals, coord_tol=coord_tol, seed=seed,
                                bench_path=bench_path, ns_path=ns_path, alg_names=alg_names, device="cpu")
                run_correlation(ns_path=ns_path, bench_path=bench_path, dim=dim,
                                features=features, alg_name=alg_names, metrics=metrics,
                                n_estimators=n_estimators, cv=cv, seed=seed, save_path="./results_cor_reg_trace")
    





        