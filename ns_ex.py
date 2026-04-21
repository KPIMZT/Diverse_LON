import torch
from ns_core import NSGeneration
from ns_io import save_ns
from ns_viz import plot_grid, plot_coverage_boxplot

def run_ex_RQ2(dim, seed, I):
    # seed = 3 # Numbers 0-9 have result data
    # dim = 2 # 2,5,10
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
    if I == "y":
        init_parents="archetype" #"archetype"(NS+) or "random"(NS)
        ns_path = f"./results_NS_trace/dim{dim}_seed{seed}.pt"
    elif I == "n":
        init_parents="random"
        ns_path = f"./results_NS_trace/dim{dim}_seed{seed}_nonarch.pt"
    plus_comma="plus"
    compute_random_baseline=True


    ns = NSGeneration(num_gauss=num_gauss, dim=dim, _mu=_mu, _lambda=_lambda, generations=generations,
                        r=r, num_samples=num_sample, feature_type=feature_type, device=device, seed=seed,
                        novelty_threshold=novelty_threshold, novelty_k=novelty_k,
                        alpha_std=alpha_std, alpha_min=alpha_min, sigma_std=sigma_std, sigma_max=sigma_max, sigma_min=sigma_min,
                        use_crossover=use_crossover, init_parents=init_parents, plus_comma=plus_comma, compute_random_baseline=compute_random_baseline)
    ns.run()
    ns.compute_coverage_all(scale=None, bins=30)
    colors = ["#0072B2",  "#009E73", "#E69F00"]
    from ns_viz import plot_grid_mpl
    if I == "y":
        plot_grid_mpl(ns, population="novelty", color_discovered= colors[2], save_path=f"./results_NS_trace/dim{dim}_seed{seed}_arch.svg")
    elif I == "n":
        plot_grid_mpl(ns, population="novelty", color_discovered= colors[1], save_path=f"./results_NS_trace/dim{dim}_seed{seed}_nonarch.svg")
        plot_grid_mpl(ns, population="random", color_discovered= colors[0], save_path=f"./results_NS_trace/dim{dim}_seed{seed}_random.svg")
    save_ns(ns, ns_path)




def run_ex_RQ2_all():
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

                ns = NSGeneration(num_gauss=num_gauss, dim=dim, _mu=_mu, _lambda=_lambda, generations=generations,
                                r=r, num_samples=num_sample, feature_type=feature_type, device=device, seed=seed,
                                novelty_threshold=novelty_threshold, novelty_k=novelty_k,
                                alpha_std=alpha_std, alpha_min=alpha_min, sigma_std=sigma_std, sigma_max=sigma_max, sigma_min=sigma_min,
                                use_crossover=use_crossover, init_parents=init_parents, plus_comma=plus_comma, compute_random_baseline=compute_random_baseline)
                ns.run()
                ns.compute_coverage_all(scale=None, bins=30)
                save_ns(ns, ns_path)





if __name__ == "__main__":
    F = input("Do you want to display a boxplot using the result data for each dimension? (No experiment will be conducted) (y/n):")
    if F == "y":
        plot_coverage_boxplot(result_dir="./results_NS")
    elif F == "n":
        F2 = input("Do you want to run NS? (Execution time will vary depending on dimensions and generations) (y/n):")
        if F2 == "y":
            F3 = input("Are you going to run all the NS experiments? That will take a lot of time. Entering 'n'　will switch to partial execution mode (y/n):")
            if F3 == "y":
                run_ex_RQ2_all()
            elif F3 == "n":
                dim = int(input("Please enter the number of dimensions. In the paper, we experimented with 2, 5, and 10 dimensions:"))
                seed = int(input("Please enter a seed. Seeds 0–9 have pre-computed result data from our experiments:"))
                I = str(input("Do you want to use archetype (for NS+)? (y/n):"))
                run_ex_RQ2(dim, seed, I)
