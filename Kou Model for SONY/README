# Bayesian Estimation of the Kou Jump-Diffusion Model — SONY

Markets move in ways that are difficult to model honestly. Returns drift, compress, and occasionally leap — discontinuously, asymmetrically, with tails far heavier than a Gaussian would suggest. The Kou (2002) jump-diffusion model was built with exactly this in mind, and this project estimates it fully, from the ground up, for Sony (`SONY`) equity returns over **2020–2026**.

The implementation is entirely self-contained. Without much use of blackbox probabilistic tools, we are trying to implement a custom Gibbs sampler and a careful conversation between the data and the prior.

---

## Why This Model

Three things consistently appear in financial return series that standard diffusion models struggle to accommodate:

- Tails are too heavy — crashes and rallies happen far more often than a Gaussian would predict
- Returns are asymmetric — bad days tend to be more extreme than good ones
- Prices jump — discrete, discontinuous moves that no continuous-path model can honestly reproduce

The Kou double-exponential model addresses all three. It lets the data tell you not just *whether* jumps occurred, but *how large* they were, *which direction* they tended to go, and *how often* to expect them - with full uncertainty attached to every answer.

---

## The Model

Log-returns decompose into a diffusive part and a jump part:
```
r_k = μΔ + σ√Δ · ε_k + J_k · Y_k

  ε_k ~ N(0, 1)
  J_k ~ Bernoulli(π)
```

When a jump occurs, its size is drawn from a double-exponential distribution:
```
       ⎧  Exp(η₁)   with probability p        (upward jump)
Y_k ~ ⎨
       ⎩ -Exp(η₂)   with probability (1 − p)  (downward jump)
```

The asymmetry between `η₁` and `η₂` is what allows the model to capture the well-known tendency of equity crashes to be sharper and faster than rallies.

---

## Bayesian Estimation

### Eliciting priors from the data

Priors are grounded directly in what the data suggests before inference begins, rather than reaching for uninformative defaults:

- `μ` and `σ` are anchored to sample moments
- Jump candidates are flagged using a volatility threshold rule
- Tail decay rates are estimated via excess-over-threshold — this isolates the exponential decay from observations that genuinely belong to the tail, rather than conflating it with the bulk of the distribution

For the jump probability, the prior is deliberately tight:
```
π ~ Beta(2, 200)
```

What we are conveying with this choice is a belief of roughly 1% daily jump probability — consistent with decades of equity market evidence — and a regularizing pressure against the model s natural tendency to over-explain variance through spurious jumps.

### Gibbs sampling

Posterior inference runs through a hand-built Gibbs sampler. At each iteration, the sampler maintains two latent variables per observation:

- `J_k` — did a jump occur on day *k*?
- `Y_k` — if so, how large was it?

These are updated in turn alongside the model parameters, all via conjugate full conditionals. Log-scale arithmetic is used throughout where numerical underflow is a risk.

---

## Getting the Details Right

Several design choices are essential to stable, well-identified inference.

### Keeping jumps honest

Without a minimum size constraint, small diffusive fluctuations tend to leak into the jump component and inflate `π`. This translates to the following constraint:
```
|Y_k| ≥ y_min = 2σ
```

is a hard boundary that keeps the diffusive and jump components doing the jobs they were each designed to do.

### Tail estimation grounded in the tails

Estimating `η₁` and `η₂` from raw sample averages conflates the bulk of the distribution with its tail. The excess-over-threshold approach instead focuses estimation where it belongs:
```
E[Y − u | Y > u]
```

What we are conveying here is a consistent estimator of the exponential decay rate, derived only from observations that exceed the threshold - the ones that actually define tail behavior.

### Respecting the continuous-time limit

The Bernoulli approximation underpinning the discrete sampler is valid only when:
```
λΔ ≪ 1
```

This condition is verified explicitly. This means that the discrete estimation procedure remains consistent with the continuous-time process it approximates - a connection that breaks silently if the condition is ignored.

---

## What the Pipeline Produces

### Exploratory diagnostics
- Price and return series with annotated jump events
- QQ plots against the fitted model

### MCMC health checks
- Trace plots and autocorrelation functions
- Effective sample size (ESS) and R-hat for all parameters

### Posterior results
- Full marginal distributions for `μ`, `σ`, `π`, `η₁`, `η₂`, `p`
- Derived quantities: annualized volatility, jump intensity `λ`, expected jump magnitudes by direction

### Predictive validation
- Simulated vs. observed return distributions
- Kurtosis comparison to verify tail fit

---

## Generated Files

| File | Description |
|------|-------------|
| `sony_empirical_priors.pkl` | Serialized prior hyperparameters |
| `sony_mcmc_results_2025.pkl` | Full MCMC chain output |
| `sony_posterior_summary_2025.csv` | Parameter posterior statistics |
| `*.png` | All diagnostic plots |

---

## Stack

| Library | Role |
|---------|------|
| `NumPy` | Sampling and array operations |
| `SciPy` | Truncated distributions, optimization |
| `Pandas` | Data ingestion and return computation |
| `Matplotlib` | Diagnostics and posterior visualizations |

---

## What This Work Shows

Fitting a jump-diffusion model is straightforward. Fitting it well requires deliberate choices at every stage. The Kou model is expressive enough to match the asymmetric tail behavior observed in real equity data — and that expressiveness demands careful prior design and identifiability constraints to function as intended.

What we are conveying through the choices made here — informative priors, minimum jump sizes, threshold-based tail estimation — is a coherent set of commitments about what the model should and should not absorb. Together, they are the difference between a posterior that reflects the data and one that reflects the model's appetite for complexity.

- The Kou model effectively captures asymmetric tail behavior in equity returns
- Bayesian estimation provides full uncertainty quantification over all parameters
- Prior specification is critical: without constraints, jump-diffusion models systematically overfit jump intensity
- Excess-over-threshold estimation of tail parameters is strongly preferred over naive moment matching
