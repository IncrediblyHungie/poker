import numpy as np, torch, xxhash
import pyspiel

class HoldemNL6:
    """
    6‑player No‑Limit Hold'em wrapper with:
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
        self._game = pyspiel.load_game(
            "universal_poker",
            {
                "numPlayers": 6,
                "blind": "100 50 0 0 0 0",  # First two are SB and BB
                "stack": "20000 20000 20000 20000 20000 20000",
                "betting": "nolimit",
            },
        )
        self._state = self._game.new_initial_state()

    # --- core wrappers --------------------------------------------------- #
    def reset(self):
        self._state = self._game.new_initial_state()
        return self.obs()

    def step(self, action_id: int):
        action = self._ACTION_SET[action_id]
        self._state.apply_action(action)
        return self.obs(), self.done(), self._state.returns()

    def obs(self):
        # Placeholder: implement observation logic
        # Example: return np.concatenate([hole_cards, board, stacks, bets, pot])
        return np.zeros(100, dtype=np.float32)  # dummy

    def done(self):
        return self._state.is_terminal()

    # --- helpers --------------------------------------------------------- #
    def legal_action_ids(self):
        legal = self._state.legal_actions()
        return [np.where(self._ACTION_SET == a)[0][0] for a in legal]
