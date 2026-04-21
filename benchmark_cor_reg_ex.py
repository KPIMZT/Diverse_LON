from benchmark_core import run_benchmark
from cor_reg_core import run_correlation
from cor_viz import plot_cor_alldim
def run_ex_RQ3(F, dim, seed, trace, run_bench):
    if F == "y":
        plot_cor_alldim(
        result_dir = "./results_cor_reg",
        dim_seeds  = {2: 3, 5: 4, 10: 9},
        alg_name   = ["CMA-ES", "DE"],
        metrics    = ["success_rate","conv_time"],
        save=False
        )
    elif F == "n":
        n_trials = 31
        max_evals = 1000*dim
        coord_tol = 1e-2
        alg_names = ["CMA-ES","DE"]
        for alg in alg_names:
            bench_path = f"./results_Benchmark_trace/dim{dim}_seed{seed}_all_{alg}.csv"
            save_path = "./results_cor_reg_trace"
            if trace == "y":
                ns_path = f"./results_NS_trace/dim{dim}_seed{seed}.pt"
            else:
                ns_path = f"./results_NS/dim{dim}_seed{seed}.pt"

            if run_bench == "y":
                bench_path = f"./results_Benchmark_trace/dim{dim}_seed{seed}_all_{alg}.csv"
                run_benchmark(n_trials=n_trials, max_evals=max_evals, coord_tol=coord_tol, seed=seed,
                                            bench_path=bench_path, ns_path=ns_path, alg_names=[alg], device="cpu")

                features = ['num_nodes', "edge_density", "num_sink", "avg_path_sinks", "avg_path_opt", "in_strength_sinks", "in_strength_opt", "global_funnel_size"]
                metrics = ["success_rate", "conv_time"]
                n_estimators= 200
                cv = 10
                run_correlation(ns_path=ns_path, bench_path=bench_path, dim=dim,
                            features=features, alg_name=[alg], metrics=metrics,
                            n_estimators=n_estimators, cv=cv, seed=seed, bins = 30, save_path= save_path)
            
            elif run_bench == "n":
                bench_path = f"./results_Benchmark/dim{dim}_seed{seed}_all_{alg}.csv"
                features = ['num_nodes', "edge_density", "num_sink", "avg_path_sinks", "avg_path_opt", "in_strength_sinks", "in_strength_opt", "global_funnel_size"]
                metrics = ["success_rate", "conv_time"]
                n_estimators= 200
                cv = 10
                run_correlation(ns_path=ns_path, bench_path=bench_path, dim=dim,
                            features=features, alg_name=[alg], metrics=metrics,
                            n_estimators=n_estimators, cv=cv, seed=seed, bins = 30, save_path= save_path)
        
        
if __name__ == "__main__":
    F = input("Do you want to display a correlation plot using the previously collected experimental data? (No experiment will be conducted) (y/n):")
    if F == "y":
        run_ex_RQ3(F, None, None, None, None)
    elif F == "n":
        trace = str(input("Do you want to use NS trace results? (y/n):"))
        dim = int(input("Please enter the number of dimensions d. In the paper, we experimented with 2, 5, and 10 dimensions:"))
        seed = int(input("Please enter a seed. Pre-computed result data is available for seeds d=2⇒3, d=5⇒4, d=10⇒9:"))
        run_bench = str(input("Do you want to run the benchmark? Entering 'n' will run correlation analysis only using existing data. (y/n):"))
        run_ex_RQ3(F, dim, seed, trace, run_bench)


