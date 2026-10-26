import numpy as np

from shipgate import sim


def test_world_is_reproducible_and_rates_valid():
    a, b = sim.make_world(3), sim.make_world(3)
    assert np.array_equal(a["pf"], b["pf"]) and np.array_equal(a["ps"], b["ps"])
    assert a["ps"].min() >= 0 and a["ps"].max() <= 1


def test_routing_everything_to_frontier_has_zero_loss():
    w = sim.make_world(5)
    assert sim.true_loss(w, 0, np.ones(w["T"], bool)) == 0.0


def test_gate_holds_a_clearly_bad_specialist():
    w = sim.make_world(7)
    w["ps"] = np.tile(np.clip(w["pf"] - 0.3, 0, 1), (5, 1))
    s, c = sim.attempts(w)
    g = sim.gate(w, s, c)
    assert not g["ship"] or g["saving"] < 0.2  # either held or nearly everything routed


def test_gate_ships_an_equal_specialist():
    w = sim.make_world(9, T=400, k=20)  # a small eval cannot show "no loss" to within 3 points; a big one can
    w["ps"] = np.tile(w["pf"], (5, 1))
    ok = sum(sim.gate(w, *sim.attempts(w))["ship"] for _ in range(20))
    assert ok >= 15
