import json

import numpy as np

from . import sim


def score(seeds, held, T=None, k=None):
    rows = []
    for s in seeds:
        w = sim.make_world(s, held, T, k)
        search, confirm = sim.attempts(w)
        n, g = sim.naive(w, search, confirm), sim.gate(w, search, confirm)
        o = sim.oracle(w, g["c"])
        rows.append((n, g, o, sim.true_loss(w, n["c"]), len(w["pf"])))
    over = sim.MARGIN
    out = {
        "worlds": len(rows),
        "naive_shipped": sum(r[0]["ship"] for r in rows),
        "naive_shipped_but_truly_over_margin": sum(r[0]["ship"] and r[0]["loss"] > over for r in rows),
        "naive_mean_true_loss_pts_when_shipped": round(100 * float(np.mean([r[0]["loss"] for r in rows if r[0]["ship"]] or [0])), 2),
        "naive_mean_saving_pct": round(100 * float(np.mean([r[0]["saving"] for r in rows])), 1),
        "gate_shipped": sum(r[1]["ship"] for r in rows),
        "gate_shipped_but_truly_over_margin": sum(r[1]["ship"] and r[1]["loss"] > over for r in rows),
        "gate_mean_true_loss_pts_when_shipped": round(100 * float(np.mean([r[1]["loss"] for r in rows if r[1]["ship"]] or [0])), 2),
        "gate_mean_saving_pct": round(100 * float(np.mean([r[1]["saving"] for r in rows])), 1),
        "gate_held_but_routed_system_was_ok": sum((not r[1]["ship"]) and r[1]["loss"] <= over for r in rows),
        "oracle_mean_saving_pct": round(100 * float(np.mean([r[2]["saving"] for r in rows])), 1),
        "specialist_alone_truly_over_margin": sum(r[3] > over for r in rows),
    }
    return out


def main():
    out = {"dev": score(range(0, 400), False), "held_out": score(range(1000, 1400), True)}
    # added after the held-out run: same held-out worlds with a bigger evaluation
    for T, k in ((100, 10), (400, 20)):
        out[f"what_if_T{T}_k{k}"] = score(range(1000, 1400), True, T, k)
    json.dump(out, open("out/result.json", "w"), indent=1)
    for k, v in out.items():
        print(k, json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
