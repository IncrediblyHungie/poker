import h5py, torch
class CFRDataset(torch.utils.data.Dataset):
    def __init__(self, blueprint_h5):
        self.f = h5py.File(blueprint_h5, "r")
        self.keys = list(self.f["strategy"].keys())

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        probs = self.f["strategy"][k][:]
        obs = _decode_state_key(k)              # deterministic decode
        target_q = probs                        # treat probs ≈ Q rank
        return torch.tensor(obs), torch.tensor(target_q)
