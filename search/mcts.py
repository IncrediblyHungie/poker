import time, math, torch
from config import CFG, device
from value_net.model import TransformerValue

class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

def select(node, c):
    best, best_u = None, -9e9
    for a, child in node.children.items():
        u = child.Q + c * math.sqrt(node.N) / (1 + child.N)
        if u > best_u:
            best, best_u = child, u
    return best

class MCTS:
    def __init__(self):
        self.vnet = TransformerValue(CFG["vnet"]).to(device)
        self.vnet.load_state_dict(torch.load(CFG["vnet"].ckpt_dir / "vnet_final.pt"))
        self.vnet.eval()

    @torch.no_grad()
    def search(self, root_state):
        root = Node(root_state.clone(), None)
        deadline = time.perf_counter() + CFG["search"].max_search_time

        while time.perf_counter() < deadline:
            node = root
            state = root_state.clone()
            while node.children:
                node = select(node, c=CFG["search"].ucb_c)
                state.apply_action(node.action)
            if not state.is_terminal():
                for a in state.legal_actions():
                    st = state.child(a)
                    node.children[a] = Node(st, node)
                node = select(node, c=CFG["search"].ucb_c)
                state.apply_action(node.action)
            leaf_feat = torch.tensor(state.obs()).unsqueeze(0).to(device)
            value = self.vnet(leaf_feat)[0].max().item()
            while node:
                node.N += 1
                node.W += value
                node.Q = node.W / node.N
                node = node.parent

        best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_action