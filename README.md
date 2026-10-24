Business Summary

This project helps us convert physical climate risk from flooding across multi-asset portfolio into useable data and the translate the probabilistic loss estimates into metrics. This metric is designed to be relevant to capital reserves and risk management. It is useful for asset managers, insurers, and risk analysts assessing whether portfolio exposures have sufficient capital to absorb severe climate related losses. This simulation is aligned with TCFD climate risk disclosure and scenario based risk management.

Using a 100,000 year Monte Carlo simulation, the portfolio produces an Annual Average Loss of ₹826,138.35 and a 95% VaR of ₹6,225,655.74. This project also solves the problem of heavy-tail by introducing dependency between assets through a Gaussian copula, which increases the 95% VaR to ₹6,478,334.10 creating a difference between correlated and independent of ₹252,678.36. The simulations worst year produces a loss of ₹14,200,000.00 which demonstrates why capital planning based solely on expected losses can underestimate tail risk. Mumbai Port contributes the largest share of portfolio AAL due to its high exposure, flood probability, and more aggressive severity tail.

Technical Summary

Monte Carlo Simulation of over 100_000 simulated years to estimate portfolio loss distributions.

Pareto fat-tailed severity distribution to capture extreme flood loss behaviour.

Gaussian copula to model correlation between assets and the events that may occur.

Cholesky decomposition to generate correlated event realizations.

Numpy vectorization for efficient large scale simulation.

Visualized Outputs:- loss distribution histogram comparing independent vs correlated models and an Exceedance Probability curve for a particular return period risk analysis.

The code demonstartes how climate risk assumptions can be translated into quantitative portfolio risk metrics suitable for capital allocation and or reservation, stress testing and physical climate risk assessment.
