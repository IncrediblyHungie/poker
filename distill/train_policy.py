import torch, tqdm, random, pickle, ray
from config import CFG, device
from distill.policy_net import PolicyNet
from play.actor import SearchActor   # produces expert trajectories

def gather_trajectory(actor_id, n_hands):
    actor = SearchActor(seed=actor_id)
    data = []
    for _ in range(n_hands):
        traj = actor.play_hand(store=True)
        data.extend(traj)
    return data

def main():
    ray.init()
    futures = [ray.remote(gather_trajectory).remote(i, 2_000) for i in range(CFG["hw"].workers)]
    expert = []
    for d in ray.get(futures): expert.extend(d)

    model = PolicyNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["distill"].lr)
    for epoch in range(CFG["distill"].epochs):
        random.shuffle(expert)
        for i in range(0, len(expert), CFG["distill"].batch):
            batch = expert[i : i + CFG["distill"].batch]
            obs = torch.tensor([b[0] for b in batch]).to(device)
            act = torch.tensor([b[1] for b in batch]).to(device)
            logits = model(obs)
            loss = torch.nn.functional.cross_entropy(logits, act)
            loss.backward()
            opt.step(); opt.zero_grad()
        torch.save(model.state_dict(), CFG["distill"].ckpt_dir / f"policy_{epoch}.pt")

if __name__ == "__main__":
    main()