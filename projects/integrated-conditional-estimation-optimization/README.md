# Integrated Conditional Estimation–Optimization

A compact contextual optimization benchmark in which a conditional scenario estimator is **selected by downstream decision cost**, not by prediction fit alone.

The problem is contextual newsvendor control. Historical contexts receive kernel weights; the optimizer chooses the weighted critical-fractile decision. A validation block selects the kernel bandwidth using realized newsvendor cost.

This is an independent research implementation inspired by the 2025–2026 integrated conditional estimation–optimization direction described in the portfolio review. It is **not** claimed as a reproduction of a particular paper.

Pipeline:

```text
context -> conditional scenario weights -> weighted stochastic optimizer -> decision cost
                              ^                                      |
                              +------ bandwidth chosen by validation-+
```

Baselines include the unconditional empirical newsvendor decision. Train, validation, and held-out evaluation seeds are separate.

Run:

```bash
pip install -e ".[dev]"
pytest -q
python -m conditional_opt.experiment
```

Natural extensions: learned conditional density models, multistage recourse, constrained conditional decisions, and end-to-end differentiable estimator selection.
