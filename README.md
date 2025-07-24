# README.md
# 🃏 Pluribus-Fast: Multiplayer No-Limit Hold'em AI Bot

Pluribus-Fast is a scalable, GPU-accelerated implementation of a multi-agent poker bot inspired by the techniques of Pluribus and DeepStack. It uses Counterfactual Regret Minimization (CFR), deep value networks, MCTS-style search, and policy distillation to train an efficient, superhuman No-Limit Texas Hold'em agent.

---

## 🚀 Features
- **6-player No-Limit Texas Hold'em** via OpenSpiel
- **CFR-based strategy training** with Ray for distributed computation
- **Transformer-based value function** using PyTorch
- **Depth-limited search (MCTS)** to approximate real-time decisions
- **Policy distillation** for fast runtime action selection
- ✅ GPU-enabled and Ray-parallelized
- ✅ Fully configurable via `config.py`

---

## 📦 Requirements
- Python 3.10+
- CUDA-compatible GPU (A100 / V100 / A6000 recommended)
- Recommended platform: Lambda Cloud, WSL2, or Ubuntu 22.04

Install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🛠️ Setup Script (Recommended)
Run everything from scratch with:
```bash
chmod +x setup.sh
./setup.sh
```
This will:
- Set up the virtual environment
- Install dependencies
- Run the full CFR → ValueNet → Distill training pipeline

---

## 🧠 Training Pipeline
You can run the full system with:
```bash
python launch.py pipeline
```
This executes:
1. `cfr/train_cfr.py`: self-play using ExternalSamplingMCCFR
2. `value_net/train_value.py`: trains the value network
3. `distill/train_policy.py`: trains the fast policy network

Run any phase individually:
```bash
python cfr/train_cfr.py
python value_net/train_value.py
python distill/train_policy.py
```

---

## 📁 Key Files
| File/Folder            | Purpose                             |
|------------------------|--------------------------------------|
| `env/holdem_env.py`   | OpenSpiel No-Limit Hold'em wrapper  |
| `env/6p_nolimit.game` | ACPC-compliant game configuration   |
| `cfr/train_cfr.py`    | CFR strategy training               |
| `value_net/model.py`  | Transformer value function          |
| `search/mcts.py`      | Depth-limited game search           |
| `distill/policy_net.py`| Distilled runtime agent             |
| `setup.sh`            | Full automation script              |

---

## 🧪 Testing OpenSpiel Integration
```bash
python -c "from env.holdem_env import HoldemNL6; env = HoldemNL6(); print(env.legal_action_ids())"
```

---

## 🧱 Troubleshooting
- If you see `SpielError: Unknown parameter 'gamedef'`, use the `gamedef_file` method instead.
- Ensure Ray workers can import `external_sampling_mccfr`.
- Run with fewer workers in `config.py` if hitting memory limits.

---

## 🤝 Contributions & Extensions
- Add real observation encodings in `obs()`
- Support 2-player variants via config
- Add exploitability evaluation or best response

---

## 📜 License
MIT License. Fork and build your own poker bot army 🤖.
