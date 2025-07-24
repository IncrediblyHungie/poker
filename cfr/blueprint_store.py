import h5py, numpy as np
class BlueprintWriter:
    def __init__(self, path):
        self.f = h5py.File(path, "w")
        self.grp = self.f.create_group("strategy")

    def write_partial(self, strat):
        # strat: dict state_key -> probs vector
        for k, v in strat.items():
            self.grp.create_dataset(k, data=np.asarray(v), compression="lzf")

    def finalize(self):
        self.f.close()
