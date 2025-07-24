# search/bankroll.py

def adjust_value(ev, bankroll, risk):
    """
    Kelly-style adjustment: f* = ev / var approximated.
    risk in [0,1] (0 = risk-neutral, 1 = full Kelly)
    """
    return ev * (1 - risk) + ev * bankroll.var() * risk
