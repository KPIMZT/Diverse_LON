from benchmark_core import run_benchmark
from cor_reg_core import run_correlation
from cor_viz import plot_cor_alldim
def run_ex_RQ3(F, dim, seed, trace):
    if F == "y":
        plot_cor_alldim(
        result_dir = "./results_cor_reg",
        dim_seeds  = {2: 3, 5: 4, 10: 9},
        alg_name   = ["CMA-ES", "DE"],
        metrics    = ["success_rate","conv_time"],
        save=True
        )
    elif F == "n":
        n_trials = 31
        max_evals = 1000*dim
        coord_tol = 1e-2
        alg_names = ["CMA-ES","DE"]
        for alg in alg_names:
            if trace == "y":
                bench_path = f"./results_Benchmark_trace/dim{dim}_seed{seed}_all_{alg}.csv"
                ns_path = f"./results_NS_trace/dim{dim}_seed{seed}.pt"
                save_path = "./results_cor_reg_trace"
            else:
                bench_path = f"./results_Benchmark/dim{dim}_seed{seed}_all_{alg}.csv"
                ns_path = f"./results_NS/dim{dim}_seed{seed}.pt"
                save_path = "./results_cor_reg"




            run_benchmark(n_trials=n_trials, max_evals=max_evals, coord_tol=coord_tol, seed=seed,
                                        bench_path=bench_path, ns_path=ns_path, alg_names=alg_names, device="cpu")
            

            features = ['num_nodes', "edge_density", "num_sink", "avg_path_sinks", "avg_path_opt", "in_strength_sinks", "in_strength_opt", "global_funnel_size"]
            metrics = ["success_rate", "conv_time"]
            n_estimators= 200
            cv = 10
            run_correlation(ns_path=ns_path, bench_path=bench_path, dim=dim,
                        features=features, alg_name=alg_names, metrics=metrics,
                        n_estimators=n_estimators, cv=cv, seed=seed, save_path= save_path)
        
        
if __name__ == "__main__":
    F = input("Do you want to display a boxplot using the result data for each dimension? (No experiment will be conducted) (y/n):")
    if F == "y":
        run_ex_RQ3(F, None, None, None)
    elif F == "n":
        dim = int(input("Please enter the number of dimensions. In the paper, we experimented with 2, 5, and 10 dimensions:"))
        seed = int(input("Please enter a seed. Numbers 0-9 have result data:"))
        trace = str(input("Do you want to use trace NS results? (y/n):"))
        run_ex_RQ3(F, dim, seed, trace)


