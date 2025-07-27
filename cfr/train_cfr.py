import ray, h5py, tqdm, numpy as np, torch
from open_spiel.python.algorithms.mccfr import ExternalSamplingSolver
from config import CFG
from env.holdem_env import HoldemNL6
from utils.seed import set_global_seed
from cfr.blueprint_store import BlueprintWriter

def worker_task(num_iter, seed):
    set_global_seed(seed)
    env = HoldemNL6()
    learner = ExternalSamplingSolver(
        env._game,  # underlying OpenSpiel game
        value_averaging=True,
    )
    learner.run(num_iter)
    return learner.average_policy().to_dict()  # Convert policy to dict format for serialization

def main():
    ray.init(num_cpus=CFG["hw"].workers)
    iters = CFG["cfr"].iterations // CFG["hw"].workers
    futures = [
        ray.remote(worker_task).remote(iters, 43 + i)
        for i in range(CFG["hw"].workers)
    ]
    strategies = ray.get(futures)

    writer = BlueprintWriter(CFG["cfr"].storage)
    for strat in strategies:
        writer.write_partial(strat)
    writer.finalize()

if __name__ == "__main__":
    main()
