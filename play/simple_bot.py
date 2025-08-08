"""Simple poker bot for 6-player No-Limit Texas Hold'em.

This script demonstrates playing a single hand of poker where
player 0 uses a very naive rule-based strategy and the remaining
players act randomly.  It uses the `HoldemNL6` environment which
wraps OpenSpiel's 6-player game configuration.
"""

from __future__ import annotations

import random
from typing import List

import sys
from pathlib import Path

# Ensure repository root is on the Python path when executing the script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from env.holdem_env import HoldemNL6


class SimplePokerBot:
    """Naive rule-based poker bot.

    The bot simply calls/checks whenever possible; otherwise it
    selects the smallest legal action.  It serves purely as an
    example and is not meant to be a strong strategy.
    """

    def act(self, env: HoldemNL6) -> int:
        legal = env.legal_action_ids()
        if 1 in legal:  # call or check available
            return 1
        return min(legal)


def random_opponent(env: HoldemNL6) -> int:
    """Randomly choose any legal action for non-bot players."""

    return random.choice(env.legal_action_ids())


def play_hand(bot: SimplePokerBot) -> List[float]:
    """Play a single hand and return the terminal rewards.

    Parameters
    ----------
    bot: SimplePokerBot
        The bot controlling player index 0.
    """

    env = HoldemNL6()
    env.reset()
    while not env.done():
        player = env._state.current_player()
        if player == 0:
            action = bot.act(env)
        else:
            action = random_opponent(env)
        env.step(action)
    return env._state.returns()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    bot = SimplePokerBot()
    returns = play_hand(bot)
    print("Hand finished. Returns:", returns)
