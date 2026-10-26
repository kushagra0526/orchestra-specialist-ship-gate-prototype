# shipgate: when is a specialist model safe to ship in place of the frontier model?

Orchestra's results page compares a fine-tuned Qwen3-8B with Sonnet on 30 operations workflows: 92.22% strict pass vs 100% over 90 attempts, 369 ms vs 1,935 ms, about 6x lower token cost (https://orchestra.ai/results/operations). The same page says the 30 tasks were used "to evaluate and select methods". Choosing among setups on the tasks you then report on, with 3 attempts per task, is where a small model can look shippable when it is not. This prototype is a simulator with known truth and a ship gate for that decision.

## The setup
A world has T recurring tasks, a frontier pass rate per task, and 5 candidate specialist configs whose true pass rates are close on most tasks, moderately worse on some and far worse (hard failures) on a few. k attempts per task are drawn for the frontier and every config, in two independent halves (search, confirm). The question: will the system lose more than 3 points of pass rate vs the frontier?
- **Naive:** pick the best config on all attempts, ship if its point estimate loses <= 3 points. Same attempts to pick and to test.
- **shipgate:** pick the config and the tasks to route to the frontier (specialist trails by >= 0.25) on the search attempts; test the routed system on the fresh confirm attempts; ship only if the 95% upper bound of its loss (resampling tasks) is <= 3 points. Savings = share of tasks not routed x (1 - 0.165), where 0.165 is the token-cost ratio from the page's own 90-attempt table.

## Results (400 simulated worlds each; truth known)
Dev: seeds 0-399 (30-60 tasks, 6 attempts). Held-out: seeds 1000-1399 (20-100 tasks, 4-10 attempts, more hard-failure tasks), committed before the first run.

| | dev naive | dev shipgate | held-out naive | held-out shipgate |
|---|---|---|---|---|
| worlds shipped | 35 | 38 | 20 | 28 |
| shipped but truly over the 3-point margin | 25 | 4 | 14 | 1 |
| true loss of what was shipped (points) | 4.14 | 1.96 | 4.25 | 1.81 |
| mean saving over all worlds | 7.3% | 6.9% | 4.2% | 4.5% |

The specialist alone was truly over the margin in 389/400 (dev) and 392/400 (held-out) worlds, so this is a harsh world: most of what a gate sees should be held. Hindsight routing (route every task that truly trails by > 0.1) would save about 60-66%.

## What it does not do well (read this)
- **It is conservative.** In 195 (dev) and 201 (held-out) worlds the gate held although the routed system was truly fine. With 20-100 tasks and 4-10 attempts the upper bound is simply too wide; the savings it certifies (4.5%) are far below the hindsight 60%.
- **A bigger evaluation helps slowly.** I added this after the held-out run, on the same held-out worlds: 100 tasks x 10 attempts: 33 shipped, 3 over margin, saving 5.8%; 400 tasks x 20 attempts: 103 shipped, 0 over margin, saving 18.0%. (Naive then ships nothing, because its point estimate sees the loss.)
- The task-routing rule (flag at 0.25 on the search attempts) is crude; a per-task shrinkage estimate would route better.
- The world, the gate and the truth are all mine. Real tasks are not independent binomials and graders are noisy. 0.165 is the cost ratio on one page's token counts, not a general number. Nothing uses Orchestra data, models or systems.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m shipgate.run
```
numpy only. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Orchestra.
