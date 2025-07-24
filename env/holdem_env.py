# env/holdem_env.py
import numpy as np
import torch
import xxhash
import pyspiel


class HoldemNL6:
    """
    6‑player No‑Limit Hold'em wrapper using OpenSpiel universal_poker
    - Expanded 7‑action discrete set
    - Observation tensor (hole, board, stacks, bets, pot)
    """

    _ACTION_SET = np.array([
        0,  # fold
        1,  # call / check
        2,  # 0.25 pot
        3,  # 0.50 pot
        4,  # pot
        5,  # 1.5 pot
        6   # all‑in
    ], dtype=np.int8)

    def __init__(self):
        # Use external .game file since inline 'gamedef' is not supported in all builds
        self._game = pyspiel.load_game("universal_poker", {
            "gamedef_file": "env/6p_nolimit.game"
        })
        self._state = self._game.new_initial_state()

    # ------------------------------------------------------------------
    def reset(self):
        self._state = self._game.new_initial_state()
        return self.obs()

    def step(self, action_id: int):
        action = self._ACTION_SET[action_id]
        self._state.apply_action(action)
        return self.obs(), self.done(), self._state.returns()

    def obs(self):
        # TODO: Replace with real 400‑dim feature vector (hole cards, board, pot, etc.)
        return np.zeros(400, dtype=np.float32)

    def done(self):
        return self._state.is_terminal()

    def legal_action_ids(self):
        legal = self._state.legal_actions()
        return [np.where(self._ACTION_SET == a)[0][0] for a in legal if a in self._ACTION_SET]
