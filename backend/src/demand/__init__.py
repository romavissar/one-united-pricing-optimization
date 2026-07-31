"""Demand estimation: survival, logistic, hedonic, and their diagnostics.

The coefficient this package exists to produce is `beta_price`, the coefficient
on `rel_price_premium` in the Cox model. Everything else either controls for
something that would otherwise contaminate it, cross-checks it, or reports what
it is worth.
"""
