"""
ns_core.py
==============
EvolutionStrategy class: init, run, coverage, deduplication.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from msg_utils import make_archetype1, make_archetype2, make_archetype5
from ns_utils import encode_genome, decode_genome, random_genome_batch, normalize_genomes, crossover_genomes, mutate_genomes, calc_coverage, compute_novelty
from ns_viz import plot_grid
from lon_core import calc_adj, adj_to_features
from lon_utils import monotonize, decompose, convert_adj_network

class NSGeneration:
    def __init__(
        self,
        num_gauss: int,
        dim: int,
        _mu: int,
        _lambda: int,
        generations: int,
        r: float,
        num_samples: int,
        feature_type: List[str],
        device: str,
        seed: int,
        novelty_threshold: float,
        novelty_k: int,
        alpha_std: float,
        alpha_min: float,
        sigma_std: float,
        sigma_max: float,
        sigma_min: float,
        use_crossover: bool,
        init_parents: str,
        plus_comma: str,
        compute_random_baseline: bool,
    ):
        self.num_gauss, self.dim, self._mu, self._lambda = num_gauss, dim, _mu, _lambda
        self.generations = generations
        self.r = r
        self.num_samples = num_samples
        self.feature_type = feature_type
        self.seed = seed
        self.device = device
        self.novelty_threshold = novelty_threshold
        self.novelty_k = novelty_k
        self.alpha_std, self.alpha_min = alpha_std, alpha_min
        self.sigma_std, self.sigma_max, self.sigma_min = sigma_std, sigma_max, sigma_min
        self.use_crossover = use_crossover
        self.init_parents = init_parents
        self.plus_comma = plus_comma
        self.compute_random_baseline = compute_random_baseline

        self._all_records: List[Dict] = []
   
        torch.manual_seed(seed)
        engine = torch.quasirandom.SobolEngine(dim, scramble=True)
        self.means = engine.draw(num_gauss).to(device)
        self.gen_rng = torch.Generator(device=device).manual_seed(seed)
        self.baseline_rng = torch.Generator(device=device).manual_seed(seed)
        self.eval_rng = torch.Generator(device=device).manual_seed(seed)
        self.lon_seed = 0
        self.rng = np.random.default_rng(seed)

    # ── helpers ──────────────────────────────────────────────────────────
    def _make_initial_genomes(self, use_all_random=False) -> torch.Tensor:
        if not use_all_random:
            a1, s1 = make_archetype1(means=self.means, sigma_c=self.sigma_max, sigma_s=self.sigma_min, 
                                     alpha_min=self.alpha_min, generator=self.rng)
            g1 = encode_genome(a1, s1).unsqueeze(0)
            a2, s2 = make_archetype2(means=self.means, sigma_val=self.sigma_min, 
                                     alpha_min=self.alpha_min, generator=self.rng)
            g2 = encode_genome(a2, s2).unsqueeze(0)
            a3, s3 = make_archetype5(means=self.means, sigma_val=self.sigma_min, generator=self.gen_rng)
            g3 = encode_genome(a3, s3).unsqueeze(0)
            rand = random_genome_batch(N=self._mu - 3, num_gauss=self.num_gauss, device=self.device,
                                       alpha_min=self.alpha_min, sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                       generator=self.gen_rng)
            genomes = torch.cat([g1, g2, g3, rand], dim=0)
        else:
            genomes = random_genome_batch(N=self._mu, num_gauss=self.num_gauss, device=self.device,
                                          alpha_min=self.alpha_min, sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                          generator=self.gen_rng)
        genomes = normalize_genomes(genomes=genomes, num_gauss=self.num_gauss,
                                    alpha_min=self.alpha_min, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        return genomes
    
    def _eval(self, genome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alphas, sigma = decode_genome(genome, self.num_gauss)
        self.eval_rng.manual_seed(self.lon_seed)
        adj, opt_alphas, opt_means = calc_adj(
            means=self.means, alphas=alphas, sigma=sigma,
            r=self.r, num_samples=self.num_samples, generator=self.eval_rng)
        features, all_features = adj_to_features(feature_type=self.feature_type, adj=adj, opt_alphas=opt_alphas, num_gauss=self.num_gauss, device=self.device)
        return features, all_features
    
    def _record(self, genome, features, all_features, gen, archived, population: str) -> None:
        rec = {
            "genome": genome.detach().clone().cpu(),
            "features": features.detach().clone().cpu(),
            "all_features": all_features.detach().clone().cpu(),
            "gen": gen, 
            "archived": archived,
            "population": population,
        }
        self._all_records.append(rec)

    def _generate_offspring(self, parents) -> torch.Tensor:
        if self.use_crossover:
            crossed = crossover_genomes(parents, self._lambda, generator=self.gen_rng)
            return mutate_genomes(genomes=crossed, num_gauss=self.num_gauss,
                                  alpha_std=self.alpha_std, sigma_std=self.sigma_std, generator=self.gen_rng)
        else:
            idx_p = torch.randint(0, self._mu, (self._lambda,), device=self.device, generator=self.gen_rng)
            return mutate_genomes(genomes=parents[idx_p], num_gauss=self.num_gauss,
                                  alpha_std=self.alpha_std, sigma_std=self.sigma_std, generator=self.gen_rng)

    # ── main loop ────────────────────────────────────────────────────────
    def run(self) -> None:
        # initial population
        use_all_random = self.init_parents != "archetype"
        genome_mu = self._make_initial_genomes(use_all_random)
        feature_mu = []
        for genome in genome_mu:
            features, all_features = self._eval(genome)
            feature_mu.append(features)
            self._record(genome=genome, features=features, all_features=all_features, gen=0, archived=True, population= "novelty")
            self._record(genome=genome, features=features, all_features=all_features, gen=0, archived=False, population= "random")
        feature_mu = torch.stack(feature_mu)
        # random baseline
        if self.compute_random_baseline:
            for gen in tqdm(range(1, self.generations + 1), desc="random"):
                rand_genomes = random_genome_batch(N=self._lambda, num_gauss=self.num_gauss, device=self.device,
                                                alpha_min=self.alpha_min, sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                                generator=self.baseline_rng)
                rand_genomes = normalize_genomes(genomes=rand_genomes, num_gauss=self.num_gauss, alpha_min=self.alpha_min,
                                                sigma_min=self.sigma_min, sigma_max=self.sigma_max)
                for genome in rand_genomes:
                    features, all_features = self._eval(genome)
                    self._record(genome=genome, features=features, all_features=all_features, gen=gen, archived=False, population= "random")

        # NS generations
        feature_archive = feature_mu.clone() 
        added_accum, genflag = 0, 0
        for gen in tqdm(range(1, self.generations + 1), desc="novelty"):
            genome_lambda = self._generate_offspring(genome_mu)
            genome_lambda = normalize_genomes(genomes=genome_lambda, num_gauss=self.num_gauss, alpha_min=self.alpha_min,
                                              sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            feature_lambda = []
            all_feature_lambda = []
            for genome in genome_lambda:
                features, all_features  = self._eval(genome)
                feature_lambda.append(features)
                all_feature_lambda.append(all_features)
            feature_lambda = torch.stack(feature_lambda)
            all_feature_lambda = torch.stack(all_feature_lambda)

            if self.plus_comma == "plus":
                features = torch.cat([feature_mu, feature_lambda])
                genomes = torch.cat([genome_mu, genome_lambda])
            elif self.plus_comma == "comma":
                features = feature_lambda
                genomes = genome_lambda

            # compute novelty
            novelty = compute_novelty(features=features, archive_features=feature_archive, k=self.novelty_k)
            top_idx = torch.topk(novelty, self._mu).indices
            genome_mu = genomes[top_idx]
            feature_mu = features[top_idx]

            new_archive = []
            offset = self._mu if self.plus_comma == "plus" else 0
            for i in range(len(genome_lambda)):
                if novelty[offset + i].item() > self.novelty_threshold:
                    self._record(genome=genome_lambda[i], features=feature_lambda[i], all_features=all_feature_lambda[i], gen=gen, archived=True, population="novelty")
                    new_archive.append(feature_lambda[i])
                    added_accum += 1
                else:
                    self._record(genome=genome_lambda[i], features=feature_lambda[i], all_features=all_feature_lambda[i], gen=gen, archived=False, population="novelty")
            if new_archive:
                feature_archive = torch.cat([feature_archive, torch.stack(new_archive)])

            if genflag == 4:
                if added_accum > 30:
                    self.novelty_threshold *= 1.05
                elif added_accum < 1:
                    self.novelty_threshold *= 0.95
                genflag, added_accum = 0, 0
            else:
                genflag += 1

    # ── coverage ─────────────────────────────────────────────────────────
    def compute_coverage_all(self, scale, bins: int):
        self.bins = bins
        self.coverage_archive: List[float] = []
        self.coverage_novelty: List[float] = []
        self.coverage_random: List[float] = []

        if scale is None:
            scale = np.ones(len(self.feature_type), dtype=np.float32)
        self.feature_scale = scale

        def _cov(feature):
            return calc_coverage(feature, scale, bins)

        for n in range(self.generations+1):
            rec = [ind for ind in self._all_records if ind['gen']<=n]
            novelty_features = torch.stack([ind['features'] for ind in rec if ind['population']=="novelty"])
            archive_features = torch.stack([ind['features'] for ind in rec if ind['population']=="novelty" and ind['archived'] is True])
            random_features = torch.stack([ind['features'] for ind in rec if ind['population']=="random"])
            self.coverage_archive.append(_cov(archive_features))
            self.coverage_novelty.append(_cov(novelty_features))
            self.coverage_random.append(_cov(random_features))
        self._rebuild_grid(scale)
        plot_grid(self, population="random",save_path=f"./results_NS/dim{self.dim}_gauss{self.num_gauss}_mu{self._mu}_lambda{self._lambda}_gen{self.generations}_seed{self.seed}_random.pdf")
        plot_grid(self, population="novelty",save_path=f"./results_NS/dim{self.dim}_gauss{self.num_gauss}_mu{self._mu}_lambda{self._lambda}_gen{self.generations}_seed{self.seed}_NS.pdf")         


    def _rebuild_grid(self, scale: np.ndarray) -> None:
        self.grid_cell_idx_archive: Dict[tuple, int] = {}
        self.grid_cell_idx_novelty: Dict[tuple, int] = {}
        self.grid_cell_idx_random: Dict[tuple, int] = {}
        self.initial_cell_keys = set()

        for idx, rec in enumerate(self._all_records):
            arr = rec["features"].numpy()
            norm = arr / (scale + 1e-12)
            idxs = np.minimum((norm * self.bins + 1e-12).astype(int), self.bins - 1)
            key = tuple(int(i) for i in idxs)
            if rec['population'] == "novelty":
                if key not in self.grid_cell_idx_novelty:
                    self.grid_cell_idx_novelty[key] = idx
                if rec['archived'] is True and key not in self.grid_cell_idx_archive:
                    self.grid_cell_idx_archive[key] = idx
            else:
                if key not in self.grid_cell_idx_random:
                    self.grid_cell_idx_random[key] = idx          
            if idx < self._mu:
                self.initial_cell_keys.add(key)


    def cell_to_instance(self, cell, population):
        cell_idx= {
            "novelty": self.grid_cell_idx_novelty,
            "archive": self.grid_cell_idx_archive,
            "random": self.grid_cell_idx_random,
        }[population]
        rec = self._all_records[cell_idx[cell]]
        genome = rec['genome']
        features = rec["features"]
        alphas, sigma = decode_genome(genome, self.num_gauss)
        eval_rng = torch.Generator(device=self.device).manual_seed(self.lon_seed)
        adj, opt_alphas, opt_means = calc_adj(
            means=self.means, alphas=alphas, sigma=sigma,
            r=self.r, num_samples=self.num_samples, generator=eval_rng)
        G = convert_adj_network(adj.cpu().numpy(), opt_alphas.cpu().numpy(), opt_means.cpu().numpy())
        monoG = monotonize(G)
        decoG = decompose(monoG)

        return genome, features, opt_alphas, opt_means, G, monoG, decoG 
