from config import device
print(f"🚀 Using device: {device}")

import fire
from cfr.train_cfr import main as cfr_run
from value_net.train_value import main as vnet_run
from distill.train_policy import main as distill_run

def full_pipeline():
    print("=== CFR training ===")
    cfr_run()
    print("=== Value net training ===")
    vnet_run()
    print("=== Distillation ===")
    distill_run()
    print("Pipeline complete ✔")

if __name__ == "__main__":
    fire.Fire({
        "pipeline": full_pipeline,
        "cfr": cfr_run,
        "vnet": vnet_run,
        "distill": distill_run,
    })