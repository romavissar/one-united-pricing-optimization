# AGENTS.md — Working rules for this repository

Read `PROJECT_BRIEF.md` and `MLS_SCHEMA.md` before writing code. This file is the
short list of rules that are easy to violate by accident.

---

## The prime directive

This project exists to estimate one number: **`β_price`**, the own-price elasticity of
demand. Every component either helps estimate it, consumes it, or reports honestly about
its uncertainty. If a change does none of those three things, it is out of scope.

The failure mode to design against: a system that produces confident, well-formatted
price recommendations that are actually just "charge the maximum" dressed up in a
dashboard. Guard against it actively.

---

## Hard rules

**1. Never hardcode an estimated parameter.**
`β_price`, floor premiums (`α`, `γ`), view premiums, absorption rates, and seasonal
multipliers are all **fitted from data**. If a number like `-1.6` or `0.058` appears as a
literal anywhere outside a test fixture, a synthetic generator, or a documented
sanity-range comment, that is a bug. Config may hold *priors and bounds*; it may not hold
answers.

**2. Never silently impute.**
Missing data gets a null and a `*_source` marker (`"reported"` / `"parsed"` / `"missing"`).
It never gets a submarket mean quietly substituted in. A wrong floor corrupts the hedonic
surface, which corrupts the price bounds, which corrupts the optimizer. Missing beats
wrong.

**3. Diagnostics fail loudly, not quietly.**
If `β(rel_price_premium) >= 0` after a fit, raise or return a hard `FAILED` diagnostic
status. Do not log a warning and continue. Do not adjust the specification to make the
sign flip. A positive price coefficient means the identification is broken and the honest
output is "this data cannot answer the question."

**4. Every optimize/simulate response carries `provenance`.**
Including `is_calibrated_on_real_data`. When it is `false`, the frontend shows a
non-dismissible banner. This is not a nice-to-have; it is the mechanism that prevents a
synthetic number from being screenshotted into a client deck.

**5. Do not add dependencies.**
The stack in `PROJECT_BRIEF.md` §2 is locked. No Streamlit, Next.js, Docker, Postgres,
Redis, ORM servers, or Electron. If a task seems to require a new library, solve it with
the existing stack or flag it and stop.

**6. Stop at the calibration gate.**
`PROJECT_BRIEF.md` §5. Build everything, verify on synthetic, write
`CALIBRATION_READY.md`, and stop. Do not autonomously fit on the real MLS export.

**7. Respect the build order.**
Each phase's acceptance criteria must pass before the next phase begins. Run the tests.
Do not build phases 1–7 and then test.

---

## Code conventions

**Python**
- 3.11+, type hints on all public functions, `snake_case`.
- Return dataclasses or `TypedDict`s from anything crossing a module boundary — not bare
  tuples or loosely-shaped dicts.
- Pure functions in `src/`; side effects (file I/O, DB writes) confined to `store.py`,
  `ingest_mls.py`, and route handlers.
- Raise domain exceptions (`IdentificationError`, `InfeasibleModelError`,
  `SchemaError`), not bare `Exception`.
- `logging`, never `print`, except in CLI `--inspect` reports where the printed report
  *is* the deliverable.
- Docstrings state units explicitly: `$/sqft`, `sqft`, `days`, `annual decimal rate`.
  Unit confusion between `$/sqft` and total price is the most likely silent numeric bug
  in this codebase.

**Frontend**
- Functional components, hooks. No class components.
- Tailwind utilities against the theme tokens in `tailwind.config.js`. No inline hex
  values, no ad-hoc CSS files.
- All numerals in tables and price columns: `font-mono tabular-nums`.
- Never `<form>` with default submission — use `onClick` handlers.
- No `localStorage` / `sessionStorage`. State lives in React state and the backend.
- Loading, empty, and error states are required for every async view, not optional polish.

**Tests**
- `pytest`. Every phase ships with its tests passing.
- Synthetic data with **planted parameters** is the primary verification tool: generate
  with a known `true_beta_price`, fit, assert recovery. Round-tripping a known truth is
  the only way to know the estimator works before real data arrives.
- Test the failure modes explicitly, not just the happy path: `β_price = 0` must push
  prices to the ceiling; infeasible constraints must return a diagnostic naming the
  binding constraint.

---

## Numerical conventions

| Quantity | Convention |
|---|---|
| Price | `$/sqft` internally, USD. Convert to total price only at presentation. |
| Area | sqft (Miami). The Bucharest config seam switches to sqm; never mix within a run. |
| Discount rate | **annual, decimal** (`0.12`, not `12`). Discount factor `1/(1+r)^(t_months/12)`. |
| Elasticity | `β` on `rel_price_premium`, which is a **ratio minus one** (`0.10` = priced 10% above submarket-month median). Interpret and label accordingly. |
| Duration | days internally; convert to months/weeks only for display. |
| Probability | `[0,1]` floats, never percentages, until formatting. |

---

## What "done" means for a phase

1. Acceptance criteria in `PROJECT_BRIEF.md` all satisfied.
2. Tests written and passing.
3. New public functions have type hints and unit-stating docstrings.
4. No hardcoded estimated parameters introduced.
5. Anything deliberately deferred is noted in `README.md` under "Known gaps" — not left
   as a silent `TODO` in the source.

---

## Known modeling simplifications — keep these documented, do not quietly "fix" them

**Independence in the objective.** The optimizer treats each unit's sale probability as
independent of the other units released in the same phase. Real units released together
compete for one finite buyer pool, so the objective **overestimates revenue for phases
that release many near-identical units**. The optional `crowding.py` correction partially
offsets this. Keep the caveat in the API response and in the report; do not remove it
because the numbers look better without it.

**Expected-value cash-flow constraint.** The per-phase cash-flow floor binds on
*expected* revenue, so realized cash flow can still breach it. Where the floor is a real
loan covenant, report the simulated breach probability and buffer the floor rather than
claiming the constraint is satisfied with certainty.

**No competitor reaction.** Competitor pricing enters as an exogenous input. Strategic
response is handled through scenario analysis, not equilibrium modeling. Say so.

**Static demand over the horizon.** Demand parameters are fitted once and held fixed
across all phases, though a real 12–18 month sales period will drift. Re-fitting between
phases using observed absorption is a future improvement, not a current capability.
