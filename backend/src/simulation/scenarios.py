"""What is uncertain about a release plan, and how those uncertainties move together.

Five channels perturb a plan. Exactly one of them is fitted:

| channel | reaches the model through | fitted? |
|---|---|---|
| `beta_price` | the coefficient on `rel_price_premium` | **yes** — `Normal(β̂, SE(β̂))` |
| `competing_listings` | the fitted coefficient on `inventory_competition` | coefficient yes, shift size no |
| `comps_drift` | rescales `rel_price_premium` at a fixed asking price | no |
| `absorption` | a direct shift in the log hazard | no |
| `completion_delay_months` | slips the construction gate | no |

The distinction matters more than it looks. `beta_price`'s spread is the real
standard error off the Cox fit, so widening it means the data was less
informative. The other four are **the user's beliefs about the future**, and
nothing in this repository estimates them. They are reported as assumptions
alongside every result for the same reason `crowding_lambda` is.

`PROJECT_BRIEF.md` §5 also asks for a mortgage-rate path. There is no macro
series joined to the listings, so a rate-to-hazard sensitivity cannot be fitted
— and inventing one would be exactly the hardcoded estimated parameter
`AGENTS.md` §1 forbids. Rates therefore enter through `absorption`, as a
log-hazard shift the user owns and the output labels.

**Sign conventions**, since half of these are easy to read backwards:
- `absorption` is additive on the log hazard. Negative is a *slower* market.
- `beta_price` is negative; more negative is more elastic.
- `comps_drift` is a proportional shift in the comp median. Positive means the
  market moved up, so a fixed asking price is *less* of a premium.
- `competing_listings` is a count shift, positive meaning more rivals.
- `completion_delay_months` is non-negative. Buildings run late.

Units: `absorption` log points; `comps_drift` decimal fraction;
`competing_listings` a count; `completion_delay_months` months.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

CHANNELS: tuple[str, ...] = (
    "beta_price",
    "absorption",
    "competing_listings",
    "comps_drift",
    "completion_delay_months",
)

NORMAL = "normal"
LHS = "lhs"

# Structural sign assumptions with round magnitudes — not estimates, and they
# travel into every result that uses them. The brief requires the channels be
# correlated rather than drawn independently: a rate shock that softens
# absorption while leaving elasticity untouched is not a scenario anyone should
# plan against. Read them as "a weak market tends to be a more price-sensitive
# market", which is the direction, and treat the numbers as adjustable.
DEFAULT_CORRELATIONS: dict[tuple[str, str], float] = {
    ("beta_price", "absorption"): 0.50,
    ("beta_price", "competing_listings"): -0.30,
    ("absorption", "competing_listings"): -0.40,
    ("absorption", "comps_drift"): 0.30,
    ("beta_price", "comps_drift"): 0.20,
}


@dataclass(frozen=True)
class ScenarioSpec:
    """The uncertainty a plan is evaluated against.

    Attributes:
        beta_price_mean: the fitted `β̂`. Must match the value baked into the
            tensor's sale probabilities, or the perturbation is measured from
            the wrong origin.
        beta_price_se: the **fitted** standard error. This is the one dispersion
            here that is evidence rather than assumption.
        absorption_log_hazard_sd: sd of a market-wide log-hazard shift.
        competing_listings_sd: sd of the shift in `inventory_competition`.
        comps_drift_sd: sd of the proportional shift in the comp median.
        completion_delay_months_sd: sd of construction slippage, clipped at
            zero — see `draw_scenarios`.
        correlation: pairwise correlations between channels; unlisted pairs are
            independent.
    """

    beta_price_mean: float
    beta_price_se: float
    absorption_log_hazard_sd: float = 0.0
    competing_listings_sd: float = 0.0
    comps_drift_sd: float = 0.0
    completion_delay_months_sd: float = 0.0
    correlation: Mapping[tuple[str, str], float] = field(
        default_factory=lambda: dict(DEFAULT_CORRELATIONS)
    )

    def __post_init__(self) -> None:
        if self.beta_price_se < 0:
            raise SchemaError(f"beta_price_se must be non-negative, got {self.beta_price_se}")
        if self.beta_price_mean >= 0:
            raise SchemaError(
                f"beta_price_mean is {self.beta_price_mean:+.4f}. A non-negative price "
                "coefficient means identification failed; simulating a plan built on it "
                "would dress that failure in percentiles. See AGENTS.md §3."
            )
        for name, value in self.standard_deviations.items():
            if value < 0:
                raise SchemaError(f"{name} standard deviation must be non-negative, got {value}")
        for (a, b), rho in self.correlation.items():
            if a not in CHANNELS or b not in CHANNELS:
                raise SchemaError(f"unknown correlation channel in {(a, b)}; known: {CHANNELS}")
            if not -1.0 <= rho <= 1.0:
                raise SchemaError(f"correlation {(a, b)} must be in [-1, 1], got {rho}")

    @property
    def standard_deviations(self) -> dict[str, float]:
        return {
            "beta_price": float(self.beta_price_se),
            "absorption": float(self.absorption_log_hazard_sd),
            "competing_listings": float(self.competing_listings_sd),
            "comps_drift": float(self.comps_drift_sd),
            "completion_delay_months": float(self.completion_delay_months_sd),
        }

    @property
    def active_channels(self) -> tuple[str, ...]:
        """Channels with non-zero dispersion. The rest are held at their mean."""
        return tuple(c for c in CHANNELS if self.standard_deviations[c] > 0)

    def correlation_matrix(self, channels: tuple[str, ...] | None = None) -> np.ndarray:
        """Correlation matrix over `channels`, defaulting to the active ones.

        Raises:
            SchemaError: when the matrix is not positive semi-definite. A
                non-PSD correlation structure describes a world that cannot
                exist; nudging it to the nearest valid one would quietly change
                the scenario the user asked for.
        """
        names = channels if channels is not None else self.active_channels
        size = len(names)
        matrix = np.eye(size)
        index = {name: i for i, name in enumerate(names)}
        for (a, b), rho in self.correlation.items():
            if a in index and b in index:
                matrix[index[a], index[b]] = matrix[index[b], index[a]] = float(rho)
        if size and np.linalg.eigvalsh(matrix).min() < -1e-8:
            raise SchemaError(
                f"The correlation matrix over {list(names)} is not positive "
                "semi-definite, so no joint distribution matches it. Reduce the "
                "magnitude of the conflicting correlations."
            )
        return matrix

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta_price_mean": self.beta_price_mean,
            "beta_price_se": self.beta_price_se,
            "standard_deviations": self.standard_deviations,
            "active_channels": list(self.active_channels),
            "correlation": {f"{a}|{b}": v for (a, b), v in self.correlation.items()},
            "fitted_channels": ["beta_price"],
            "assumption_channels": [c for c in self.active_channels if c != "beta_price"],
        }


@dataclass
class ScenarioDraws:
    """One array per channel, each of length `n_draws`."""

    beta_price: np.ndarray
    absorption: np.ndarray
    competing_listings: np.ndarray
    comps_drift: np.ndarray
    completion_delay_months: np.ndarray
    spec: ScenarioSpec
    method: str
    seed: int
    notes: list[str] = field(default_factory=list)

    @property
    def n_draws(self) -> int:
        return int(self.beta_price.shape[0])

    def channel(self, name: str) -> np.ndarray:
        if name not in CHANNELS:
            raise SchemaError(f"unknown channel {name!r}; known: {CHANNELS}")
        return getattr(self, name)  # type: ignore[no-any-return]

    def realized_correlation(self) -> dict[str, float]:
        """Sample correlations between active channels, for verification.

        Worth reading: LHS stratifies each margin and then gets rotated by the
        Cholesky factor, so the realized correlation is close to the target but
        not equal to it, and clipping the delay channel pulls its correlations
        toward zero.
        """
        active = [c for c in self.spec.active_channels]
        out: dict[str, float] = {}
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                x, y = self.channel(a), self.channel(b)
                if x.std() > 0 and y.std() > 0:
                    out[f"{a}|{b}"] = float(np.corrcoef(x, y)[0, 1])
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_draws": self.n_draws,
            "method": self.method,
            "seed": self.seed,
            "spec": self.spec.as_dict(),
            "realized_correlation": self.realized_correlation(),
            "notes": self.notes,
        }


def _latin_hypercube(n_draws: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified uniforms, one stratum per draw per dimension."""
    grid = np.tile(np.arange(n_draws, dtype="float64"), (n_dim, 1))
    shuffled = rng.permuted(grid, axis=1)
    return ((shuffled + rng.random((n_dim, n_draws))) / n_draws).T


def draw_scenarios(
    spec: ScenarioSpec, n_draws: int, *, seed: int = 0, method: str = NORMAL
) -> ScenarioDraws:
    """Sample correlated parameter scenarios.

    Args:
        spec: dispersions and correlations.
        n_draws: number of scenarios.
        seed: makes the draw reproducible. A revenue distribution that changes
            between runs is not a result anyone can act on.
        method: `"normal"` for plain Gaussian draws, `"lhs"` for a Latin
            hypercube. LHS covers the space more evenly at small sample sizes,
            which is what the re-solve mode needs from ~100 scenarios.

    Returns:
        `ScenarioDraws`. Inactive channels come back as arrays of zeros, so
        downstream arithmetic needs no special cases.

    Raises:
        SchemaError: on a non-positive draw count or an unknown method.
    """
    if n_draws < 1:
        raise SchemaError(f"n_draws must be at least 1, got {n_draws}")
    if method not in {NORMAL, LHS}:
        raise SchemaError(f"method must be {NORMAL!r} or {LHS!r}, got {method!r}")

    rng = np.random.default_rng(seed)
    active = spec.active_channels
    notes: list[str] = []
    values = {c: np.zeros(n_draws, dtype="float64") for c in CHANNELS}

    if active:
        chol = np.linalg.cholesky(spec.correlation_matrix(active))
        if method == LHS:
            standard = norm.ppf(_latin_hypercube(n_draws, len(active), rng))
            notes.append(
                "Latin hypercube draws are rotated by the Cholesky factor to induce "
                "correlation, which partly undoes the stratification. Coverage is still "
                "far better than independent sampling at this draw count."
            )
        else:
            standard = rng.standard_normal((n_draws, len(active)))
        correlated = standard @ chol.T
        for position, name in enumerate(active):
            values[name] = correlated[:, position] * spec.standard_deviations[name]

    values["beta_price"] = values["beta_price"] + spec.beta_price_mean

    if spec.completion_delay_months_sd > 0:
        clipped = int((values["completion_delay_months"] < 0).sum())
        values["completion_delay_months"] = np.clip(values["completion_delay_months"], 0.0, None)
        notes.append(
            f"construction delay clipped at zero in {clipped} of {n_draws} draws — "
            "buildings run late, not early. The realized mean delay is therefore "
            "positive even though the channel is centred on zero, and its correlations "
            "are pulled toward zero by the clip."
        )

    positive = int((values["beta_price"] >= 0).sum())
    if positive:
        notes.append(
            f"{positive} of {n_draws} draws put beta_price at or above zero "
            f"({positive / n_draws:.1%}). Those are worlds where price does not deter "
            "buyers and the plan's revenue is bounded only by the price ceiling. A "
            "large share here means the fit was too imprecise to plan against."
        )

    logger.info(
        "Drew %d %s scenarios over %s (seed=%d)", n_draws, method, list(active), seed
    )
    return ScenarioDraws(
        beta_price=values["beta_price"],
        absorption=values["absorption"],
        competing_listings=values["competing_listings"],
        comps_drift=values["comps_drift"],
        completion_delay_months=values["completion_delay_months"],
        spec=spec,
        method=method,
        seed=seed,
        notes=notes,
    )


def spec_from_fit(
    beta_price: float, beta_price_se: float, **assumptions: Any
) -> ScenarioSpec:
    """Build a spec from a fitted coefficient plus explicit user assumptions.

    A thin constructor whose only job is to make the call site read as what it
    is: the elasticity dispersion comes from the fit, everything else is
    supplied by whoever is running the scenario.
    """
    return ScenarioSpec(
        beta_price_mean=float(beta_price), beta_price_se=float(beta_price_se), **assumptions
    )
