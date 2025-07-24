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
        # Inline ACPC‑style GAMEDEF. **First token after GAMEDEF must be “nolimit”.**
        # OpenSpiel requires exactly six blind values (one per seat). Here we
        # set SB = 100, BB = 50, and 0 for the remaining four players.
        self._game = pyspiel.load_game("universal_poker", {
            "gamedef": (
                "GAMEDEF\n"
                "nolimit\n"
                "numPlayers   = 6\n"
                "numRounds    = 4\n"
                "blind        = 100 50 0 0 0 0\n"  # <-- six integers
                "firstPlayer  = 2\n"
                "maxRaises    = 255\n"
                "raiseSize    = 0\n"      # ignored for NLH
                "numSuits     = 4\n"
                "numRanks     = 13\n"
                "numHoleCards = 2\n"
                "numBoardCards= 0 3 1 1\n"  # flop‑turn‑river
                "stack        = 20000 20000 20000 20000 20000 20000\n"
            )
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