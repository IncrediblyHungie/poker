from pathlib import Path
from dataclasses import dataclass
import torch

ROOT = Path(__file__).parent

@dataclass
class Hardware:
    gpus: int = 1               # 1 on A100, 2 on A6000, 8 on V100 nodes
    workers: int = 28           # vCPUs you allocate to Ray actors
    ram_gb: int = 200

@dataclass
class CFRCfg:
    iterations: int = 3_000_000
    abstraction: str = "action7"       # 7‑way action set incl. bankroll
    save_every: int = 50_000
    seed: int = 42
    storage: Path = ROOT / "blueprints" / "nl6_action7.h5"

@dataclass
class ValueNetCfg:
    epochs: int = 4
    batch_size: int = 4096
    lr: float = 3e-4
    model_dim: int = 512
    heads: int = 8
    depth: int = 6
    fp16: bool = True
    ckpt_dir: Path = ROOT / "value_net_ckpts"

@dataclass
class SearchCfg:
    max_search_time: float = 4.5        # seconds, leaving 0.5 s buffer
    max_depth: int = 2
    rollout_batch: int = 256            # states evaluated in parallel
    ucb_c: float = 1.4

@dataclass
class DistillCfg:
    epochs: int = 3
    batch: int = 8192
    lr: float = 1e-4
    ckpt_dir: Path = ROOT / "policy_ckpts"

CFG = dict(
    hw=Hardware(),
    cfr=CFRCfg(),
    vnet=ValueNetCfg(),
    search=SearchCfg(),
    distill=DistillCfg(),
)

# Automatically detect and set the best device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")