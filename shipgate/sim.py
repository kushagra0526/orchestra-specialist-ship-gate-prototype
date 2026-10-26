"""Specialist-vs-frontier ship decision on a task set with known truth.

World: T recurring tasks. Frontier pass rate p_f[t]. M candidate specialist configs with true pass rates p_s[c, t]
(most tasks near-equal, some moderately worse, a few hard failures). k attempts per task for the frontier and
for every config; attempts are Binomial draws from the true rates.
"""
import numpy as np

MARGIN = 0.03        # ship if the system loses at most 3 points of pass rate vs frontier
FLAG = 0.25          # route a task to the frontier if the specialist trails by this much on the search attempts
COST_RATIO = 0.165   # specialist token cost / frontier token cost (from Orchestra's own 90-attempt table: $0.0066 / $0.0400)


def make_world(seed, held_out=False, T=None, k=None):
    rng = np.random.default_rng(seed)
    T0 = int(rng.choice([20, 45, 100] if held_out else [30, 60]))
    k0 = int(rng.choice([4, 10] if held_out else [6]))
    T, k = T or T0, k or k0
    M = 5
    pf = rng.uniform(0.88, 1.0, T)
    u = rng.random(T)
    hard, mod = (0.12, 0.25) if held_out else (0.08, 0.20)
    d = np.where(u < hard, rng.uniform(0.4, 0.9, T) * pf,
                 np.where(u < hard + mod, rng.uniform(0.04, 0.2, T), rng.normal(0.0, 0.015, T)))
    e = np.array([0.0, 0.01, 0.02, 0.04, 0.06])
    D = d[None, :] * rng.uniform(0.6, 1.4, (M, T)) + e[:, None]
    ps = np.clip(pf[None, :] - D, 0.0, 1.0)
    return {"T": T, "k": k, "pf": pf, "ps": ps, "rng": rng}


def draw(w):
    rng, k = w["rng"], w["k"]
    f = rng.binomial(k, w["pf"][None, :], (1, w["T"])) / k
    s = rng.binomial(k, w["ps"], w["ps"].shape) / k
    return f[0], s


def attempts(w):
    """Two independent halves of attempts (search, confirm), each k attempts per task."""
    return draw(w), draw(w)


def true_loss(w, c, routed=None):
    p = w["ps"][c] if routed is None else np.where(routed, w["pf"], w["ps"][c])
    return float(np.mean(w["pf"] - p))


def naive(w, search, confirm):
    """Pick the best config on all attempts, ship if the point estimate loses <= MARGIN. Same attempts for pick and test."""
    f = (search[0] + confirm[0]) / 2
    s = (search[1] + confirm[1]) / 2
    c = int(np.argmax(s.mean(1)))
    ship = f.mean() - s[c].mean() <= MARGIN
    return {"ship": bool(ship), "c": c, "routed": None, "loss": true_loss(w, c), "saving": 1 - COST_RATIO if ship else 0.0}


def gate(w, search, confirm, B=1000):
    """Pick the config and the tasks to route on the search attempts; test the routed system on fresh attempts;
    ship only if the 95% upper bound (resampling tasks) of its loss is <= MARGIN."""
    fs, ss = search
    fc, sc = confirm
    c = int(np.argmax(ss.mean(1)))
    routed = (fs - ss[c]) >= FLAG
    diff = fc - np.where(routed, fc, sc[c])
    idx = w["rng"].integers(0, w["T"], (B, w["T"]))
    upper = np.quantile(diff[idx].mean(1), 0.95)
    ship = upper <= MARGIN
    saving = (1 - COST_RATIO) * (1 - routed.mean()) if ship else 0.0
    return {"ship": bool(ship), "c": c, "routed": routed, "loss": true_loss(w, c, routed), "saving": float(saving), "upper": float(upper)}


def oracle(w, c):
    """Hindsight: route every task where the config truly trails by > 0.1."""
    routed = (w["pf"] - w["ps"][c]) > 0.1
    return {"routed": routed, "loss": true_loss(w, c, routed), "saving": (1 - COST_RATIO) * (1 - routed.mean())}
