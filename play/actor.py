import torch
from search.mcts import MCTS
from distill.policy_net import PolicyNet
from config import CFG
from env.holdem_env import HoldemNL6

class SearchActor:
    def __init__(self, seed=0):
        self.env = HoldemNL6()
        torch.manual_seed(seed)
        self.mcts = MCTS()
        self.fallback = PolicyNet()
        self.fallback.load_state_dict(torch.load(_latest(CFG["distill"].ckpt_dir)))
        self.fallback.cuda().eval()

    def play_hand(self, store=False):
        obs = self.env.reset()
        traj = []
        while not self.env.done():
            if self.env._state.cur_player() == 0:     # we are player 0
                try:
                    act = self.mcts.search(self.env._state)
                except TimeoutError:
                    act = self.fallback(torch.tensor(obs).unsqueeze(0).cuda()).argmax().item()
            else:
                act = _simple_rule_bot(self.env)      # random opponent
            if store and self.env._state.cur_player() == 0:
                traj.append((obs, act))
            obs, *_ = self.env.step(act)
        return traj
