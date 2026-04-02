"""
Sony Corporation (SONY) -- Full Bayesian Analysis of the Kou Jump-Diffusion Model
                           [ 1-YEAR WINDOW: 2020-01-01 to 2026-01-01 ]
==================================================================================
PART 1 -- Empirical Prior Estimation
--------------------------------------
Model: Kou (2002) Double-Exponential Jump-Diffusion
       dS = mu*S dt + sigma*S dW + S d(sum Y_i)
       Y_i ~ p*Exp(eta1) + (1-p)*Exp(eta2)  [double-exponential jumps]

PART 2 -- Bayesian MCMC (Gibbs Sampler / Bernoulli discretisation)
-----------------------------------------
    r_k = mu*Delta + sigma*sqrt(Delta)*eps_k + J_k*Y_k
    J_k  ~ Bernoulli(pi)
    Y_k  ~ p*Exp(eta1)*1{y>=y_min} + (1-p)*Exp(eta2)*1{y<=-y_min}

The Bernoulli discretisation is PRESERVED (NOT replaced by Poisson).
With these fixes posterior lambda stays well inside lambda*Delta << 1.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.special import logsumexp
import pickle

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG  --  Only edit this block
# =============================================================================

EXCEL_PATH        = r"C:\Users\zelya\OneDrive - Concordia University - Canada\Documents\Concordia\Graduate Studies\Courses\Year 2\Levy\SONY.xlsx"

DATE_START        = '2020-01-02'   # inclusive
DATE_END          = '2026-02-01'   # exclusive

# ---- Prior elicitation tuning -----------------------------------------------
JUMP_SIGMA        = 3      # |return| > JUMP_SIGMA * sigma  => classified as jump (prior)
PRIOR_STRENGTH_P  = 10     # pseudo-observations for p  Beta prior
A_SIGMA           = 5.0    # IG shape for sigma^2
A_ETA             = 3.0    # Gamma shape for eta1, eta2
SAVE_PRIORS       = True

# FIX 3: Strongly informative pi prior centred at ~1%/day (~2.5 jumps/yr).
# Beta(a, b) => E[pi] = a/(a+b).  Beta(2, 200) => E[pi] = 0.0099.
# Realistic equity range: lambda = 3-15 jumps/year.
PI_PRIOR_A        = 2.0    # FIX 3
PI_PRIOR_B        = 200.0  # FIX 3

# Minimum absolute jump size in units of daily sigma.
# Jumps with |Y| < y_min are absorbed into diffusion.
# Recommended: 1.5 -- 2.5.  Default 2.0 matches the "2-sigma rule".
Y_MIN_SIGMA       = 2.0    # FIX 2

# ---- MCMC settings ----------------------------------------------------------
N_ITER            = 15_000
N_BURNIN          = 5_000
N_THIN            = 5
DELTA             = 1 / 252
TRUNC_ETA1        = True    # enforce eta1 > 1  (risk-neutral integrability)
RANDOM_SEED       = 42
PLOT_RESULTS      = True
SAVE_RESULTS      = True

N_KEEP = (N_ITER - N_BURNIN) // N_THIN


# =============================================================================
# PART 1 -- DATA LAYER
# =============================================================================

def fetch_data(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=0)
    raw.columns = raw.columns.str.strip()

    col_map = {}
    for col in raw.columns:
        low = col.lower().replace(' ', '_').replace('.', '')
        if low == 'date':
            col_map[col] = 'date'
        elif low in ('adj_close', 'adjusted_close', 'adj_close_'):
            col_map[col] = 'adjusted_close'
        elif low == 'close':
            col_map[col] = 'close'
        elif low in ('open', 'high', 'low', 'volume'):
            col_map[col] = low
    raw = raw.rename(columns=col_map)

    if 'date' not in raw.columns:
        raise ValueError("No 'Date' column found in Excel file.")
    if 'adjusted_close' not in raw.columns:
        raise ValueError("No 'Adj Close' column found. Columns: "
                         + str(list(raw.columns)))

    raw['date']           = pd.to_datetime(raw['date'], errors='coerce')
    raw['adjusted_close'] = pd.to_numeric(raw['adjusted_close'], errors='coerce')

    n_before  = len(raw)
    raw       = raw.dropna(subset=['date', 'adjusted_close'])
    n_dropped = n_before - len(raw)
    if n_dropped:
        print(f"[CLEAN] Dropped {n_dropped} non-numeric / dividend rows")

    raw = raw.sort_values('date').set_index('date')
    print(f"[OK] Loaded {len(raw)} trading days "
          f"({raw.index[0].date()} to {raw.index[-1].date()})")
    return raw[['adjusted_close']]


# =============================================================================
# PART 1 -- PRIOR ELICITATION
# =============================================================================

def compute_priors(returns: pd.Series) -> dict:
    """
    Elicit conjugate priors from the empirical return distribution.

    FIX 3 applied: pi prior uses PI_PRIOR_A / PI_PRIOR_B (not data-driven).
    FIX 4 applied: eta hats estimated from excess over threshold, not raw level.
    """
    n    = len(returns)
    mu   = returns.mean()
    sig  = returns.std(ddof=1)
    se   = sig / np.sqrt(n)
    kurt = returns.kurtosis()

    # mu prior: Normal(mean, variance)
    mu_prior = (mu, (2 * se) ** 2)

    # sigma^2 prior: IG(a, b),  E[sigma^2] = b/(a-1)
    b_sig        = sig**2 * (A_SIGMA - 1)
    sigma2_prior = (A_SIGMA, b_sig)

    #  pi prior 
    pi_hat   = float(np.clip(0.02 + 0.01 * max(kurt, 0), 0.01, 0.15))  # for info only
    a_pi     = PI_PRIOR_A
    b_pi     = PI_PRIOR_B
    pi_prior = (a_pi, b_pi)
    # -------------------------------------------------------------------------

    # p prior
    thresh  = JUMP_SIGMA * sig
    n_pos   = (returns >  thresh).sum()
    n_neg   = (returns < -thresh).sum()
    p_hat   = n_pos / max(n_pos + n_neg, 1)
    a_p     = 1 + PRIOR_STRENGTH_P * p_hat
    b_p     = 1 + PRIOR_STRENGTH_P * (1 - p_hat)
    p_prior = (a_p, b_p)

    # ---- correct exponential tail estimators -------------------------
    # Memoryless property: E[Y - thresh | Y > thresh] = 1/eta.
    # Using E[Y] (which includes the threshold) overestimates eta and
    # encourages the sampler to produce near-zero jumps.
    pos_jumps = returns[returns >  thresh]
    neg_jumps = returns[returns < -thresh]

    if len(pos_jumps) > 5:
        excess_pos = pos_jumps.values - thresh
        excess_pos = excess_pos[excess_pos > 1e-10]
        eta1_hat   = (1.0 / np.mean(excess_pos)
                      if len(excess_pos) > 0 else 1.0 / (3 * sig))
    else:
        eta1_hat = 1.0 / (3 * sig)

    if len(neg_jumps) > 5:
        excess_neg = np.abs(neg_jumps.values) - thresh
        excess_neg = excess_neg[excess_neg > 1e-10]
        eta2_hat   = (1.0 / np.mean(excess_neg)
                      if len(excess_neg) > 0 else 1.0 / (3 * sig))
    else:
        eta2_hat = 1.0 / (3 * sig)
    # -------------------------------------------------------------------------

    eta1_prior = (A_ETA, A_ETA / eta1_hat)
    eta2_prior = (A_ETA, A_ETA / eta2_hat)

    return dict(
        mu     = mu_prior,
        sigma2 = sigma2_prior,
        pi     = pi_prior,
        p      = p_prior,
        eta1   = eta1_prior,
        eta2   = eta2_prior,
        _diagnostics = dict(
            n               = n,
            years           = n / 252,
            mu_annual       = mu  * 252,
            sigma_annual    = sig * np.sqrt(252),
            excess_kurtosis = kurt,
            jump_threshold  = thresh,
            n_pos_jumps     = int(n_pos),
            n_neg_jumps     = int(n_neg),
            pi_hat          = pi_hat,
            p_hat           = float(p_hat),
            eta1_hat        = float(eta1_hat),
            eta2_hat        = float(eta2_hat),
            daily_sigma     = float(sig),
        )
    )


# =============================================================================
# PART 1 -- PRIOR PRINTER
# =============================================================================

def print_priors(priors: dict):
    d   = priors['_diagnostics']
    SEP = "-" * 60

    def bm(a, b):  return a / (a + b)
    def gm(a, b):  return a / b
    def igm(a, b): return b / (a - 1)

    print()
    print("=" * 60)
    print("  SONY -- Empirical Priors for Levy Jump-Diffusion BDA")
    print(f"  {d['n']} daily returns  |  {d['years']:.1f} years  |  "
          f"sigma_ann={d['sigma_annual']:.2%}  |  ExKurt={d['excess_kurtosis']:.3f}")
    print("=" * 60)

    rows = [
        ("mu  (drift)",
         "Normal",
         priors['mu'],
         f"mean={priors['mu'][0]:.5f}  annual ~ {d['mu_annual']:.2%}"),
        ("sigma^2 (diffusion var)",
         "Inv-Gamma",
         priors['sigma2'],
         f"E[sigma^2]={igm(*priors['sigma2']):.6f}  sigma ~ {d['sigma_annual']:.2%}/yr"),
        ("pi  (jump prob/day)  [FIX3]",
         "Beta",
         priors['pi'],
         f"E[pi]={bm(*priors['pi']):.4f}  ~ {bm(*priors['pi'])*252:.1f} jumps/yr"),
        ("p   (P[jump > 0])",
         "Beta",
         priors['p'],
         f"E[p]={bm(*priors['p']):.4f}  "
         f"({d['n_pos_jumps']} up / {d['n_neg_jumps']} down extreme obs)"),
        ("eta1 (pos decay rate)  [FIX4]",
         "Gamma",
         priors['eta1'],
         f"E[eta1]={gm(*priors['eta1']):.2f}  "
         f"=> E[excess jump]={1/gm(*priors['eta1']):.4f}"),
        ("eta2 (neg decay rate)  [FIX4]",
         "Gamma",
         priors['eta2'],
         f"E[eta2]={gm(*priors['eta2']):.2f}  "
         f"=> E[|excess jump|]={1/gm(*priors['eta2']):.4f}"),
    ]

    for name, dist, params, note in rows:
        print()
        print(f"  {name}")
        print(f"  {SEP}")
        print(f"    Distribution  : {dist}{params}")
        print(f"    Interpretation: {note}")

    y_min_val = Y_MIN_SIGMA * d['daily_sigma']
    print()
    print(f"  [FIX2] Minimum jump |Y| >= {Y_MIN_SIGMA} * sigma_daily"
          f" = {y_min_val:.5f}  (enforced in MCMC)")
    print("=" * 60)
    print()


# =============================================================================
# PART 1 -- PRIOR DIAGNOSTIC PLOT
# =============================================================================

def plot_diagnostics(df: pd.DataFrame, returns: pd.Series, priors: dict):
    d      = priors['_diagnostics']
    sig    = returns.std()
    thresh = d['jump_threshold']

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax4, ax5, ax6 = [fig.add_subplot(gs[1, i]) for i in range(3)]

    ax1.plot(df.index, df['adjusted_close'], lw=1.2, color='steelblue')
    ax1.axvline(pd.Timestamp('2020-03-01'), color='crimson', ls='--',
                alpha=0.7, label='COVID crash')
    ax1.set_title('Price History', fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.plot(returns.index, returns, lw=0.6, alpha=0.6, color='dimgray')
    j_ret = returns[returns.abs() > thresh]
    ax2.scatter(j_ret.index, j_ret,
                c=['green' if r > 0 else 'red' for r in j_ret],
                s=25, zorder=5, label=f'|r|>{JUMP_SIGMA}s  (n={len(j_ret)})')
    for h in [thresh, -thresh]:
        ax2.axhline(h, color='darkorange', ls='--', lw=0.8, alpha=0.8)
    ax2.axhline(0, color='k', lw=0.4)
    ax2.set_title('Log Returns + Detected Jumps', fontweight='bold')
    ax2.set_ylabel('Log Return')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    ax3.hist(returns, bins=70, density=True, color='steelblue',
             alpha=0.65, edgecolor='white', linewidth=0.3)
    x = np.linspace(returns.min(), returns.max(), 400)
    ax3.plot(x, stats.norm.pdf(x, returns.mean(), sig), 'r--', lw=2, label='Normal fit')
    ax3.set_title(f'Return Distribution  (ExKurt={d["excess_kurtosis"]:.2f})',
                  fontweight='bold')
    ax3.set_xlabel('Log Return')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    stats.probplot(returns, dist='norm', plot=ax4)
    ax4.set_title('Normal Q-Q  (tails = jumps)', fontweight='bold')
    ax4.grid(alpha=0.25)

    if len(j_ret):
        colors = ['green' if r > 0 else 'crimson' for r in j_ret]
        ax5.bar(range(len(j_ret)), j_ret.values, color=colors, alpha=0.8)
        ax5.axhline(0, color='k', lw=0.5)
        ax5.set_title(f'Jump Magnitudes  ({len(j_ret)} events)', fontweight='bold')
        ax5.set_xlabel('Event index')
        ax5.set_ylabel('Log Return')
        ax5.grid(alpha=0.25)

    ax6.axis('off')

    def bm(a, b):  return f"{a/(a+b):.4f}"
    def gm(a, b):  return f"{a/b:.2f}"
    def igm(a, b): return f"{b/(a-1):.6f}"

    y_min_val = Y_MIN_SIGMA * d['daily_sigma']
    summary = (
        "ELICITED PRIORS  \n"
        "---------------------------------\n"
        f"  mu     ~ N({priors['mu'][0]:.5f}, {priors['mu'][1]:.5f})\n"
        f"  s^2    ~ IG({priors['sigma2'][0]:.1f}, {priors['sigma2'][1]:.5f})\n"
        f"             E[s^2] = {igm(*priors['sigma2'])}\n"
        f"  pi     ~ Beta({priors['pi'][0]:.2f},{priors['pi'][1]:.2f})  [FIX3]\n"
        f"             E[pi] = {bm(*priors['pi'])}  (~{float(bm(*priors['pi']))*252:.1f}/yr)\n"
        f"  p      ~ Beta({priors['p'][0]:.2f}, {priors['p'][1]:.2f})\n"
        f"             E[p]  = {bm(*priors['p'])}\n"
        f"  eta1   ~ Gamma({priors['eta1'][0]:.1f}, {priors['eta1'][1]:.3f})  [FIX4]\n"
        f"             E[eta1]= {gm(*priors['eta1'])}\n"
        f"  eta2   ~ Gamma({priors['eta2'][0]:.1f}, {priors['eta2'][1]:.3f})  [FIX4]\n"
        f"             E[eta2]= {gm(*priors['eta2'])}\n"
        "---------------------------------\n"
        f"  Jumps : {d['n_pos_jumps']} up / {d['n_neg_jumps']} down\n"
        f"  Thresh: {JUMP_SIGMA}s = {thresh:.5f}\n"
        f"  y_min : {Y_MIN_SIGMA}s = {y_min_val:.5f}  [FIX2]"
    )
    ax6.text(0.05, 0.97, summary, transform=ax6.transAxes, fontsize=9.0,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', fc='#FFFBEA', ec='#CCAA44', lw=1.2))

    fig.suptitle('Sony Corp -- Empirical Prior Diagnostics',
                 fontsize=12, fontweight='bold', y=1.01)

    out_dir   = os.path.dirname(os.path.abspath(EXCEL_PATH))
    plot_path = os.path.join(out_dir, 'sony_levy_diagnostics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[SAVED] Diagnostic plot -> {plot_path}")


# =============================================================================
# PART 2 -- PROGRESS BAR
# =============================================================================

class _ProgressBar:
    def __init__(self, total, width=50):
        self.total = total
        self.width = width
        self.start = time.time()

    def update(self, i):
        frac    = (i + 1) / self.total
        filled  = int(self.width * frac)
        bar     = '#' * filled + '-' * (self.width - filled)
        elapsed = time.time() - self.start
        eta     = elapsed / frac * (1 - frac) if frac > 0 else 0
        print(f"\r  [{bar}] {frac*100:5.1f}%  "
              f"elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s", end='', flush=True)

    def close(self):
        print()


# =============================================================================
# PART 2 -- TRUNCATED-NORMAL SAMPLER
# =============================================================================

def _sample_trunc_normal(mean: float, std: float,
                          lower: float = -np.inf,
                          upper: float =  np.inf) -> float:
    """
    Draw from N(mean, std^2) truncated to [lower, upper] via inverse-CDF.
    Degenerate guard: if the interval has < 1e-14 probability mass (very
    unlikely given realistic parameters), returns the nearest boundary rather
    than letting ppf blow up.
    """
    a = stats.norm.cdf((lower - mean) / std) if lower > -np.inf else 0.0
    b = stats.norm.cdf((upper - mean) / std) if upper <  np.inf else 1.0

    if b - a < 1e-14:                       # degenerate interval guard
        return lower if np.isfinite(lower) else upper

    u = np.random.uniform(a, b)
    u = np.clip(u, 1e-15, 1 - 1e-15)
    return mean + std * stats.norm.ppf(u)


# =============================================================================
# PART 2 -- GIBBS SAMPLER
# =============================================================================

class KouGibbsSampler:
    """
    Gibbs sampler for the Kou double-exponential jump-diffusion.
    """

    def __init__(self, returns: np.ndarray, priors: dict, delta: float = DELTA):
        self.r     = returns.flatten()
        self.T     = len(self.r)
        self.delta = delta

        # minimum absolute jump size (fixed across MCMC iterations)
        sig_hat    = float(priors['_diagnostics']['daily_sigma'])
        self.y_min = Y_MIN_SIGMA * sig_hat
        print(f"  [FIX2] y_min = {Y_MIN_SIGMA} * {sig_hat:.6f} = {self.y_min:.6f}")

        # Prior hyperparameters
        self.m0,   self.v0   = priors['mu']
        self.a_s,  self.b_s  = priors['sigma2']
        self.a_pi, self.b_pi = priors['pi']
        self.a_p,  self.b_p  = priors['p']
        self.a_e1, self.b_e1 = priors['eta1']
        self.a_e2, self.b_e2 = priors['eta2']
        self.priors = priors

        # Storage
        self.mu_s     = np.empty(N_KEEP)
        self.sig2_s   = np.empty(N_KEEP)
        self.pi_s     = np.empty(N_KEEP)
        self.p_s      = np.empty(N_KEEP)
        self.eta1_s   = np.empty(N_KEEP)
        self.eta2_s   = np.empty(N_KEEP)
        self.J_s      = np.empty((N_KEEP, self.T), dtype=bool)
        self.Y_s      = np.empty((N_KEEP, self.T))
        self.loglik_s = np.empty(N_KEEP)

        # Initialise state
        np.random.seed(RANDOM_SEED)
        self.mu   = float(self.m0)
        self.sig2 = float(self.b_s / max(self.a_s - 1, 1e-6))
        self.pi   = float(self.a_pi / (self.a_pi + self.b_pi))
        self.p    = float(self.a_p  / (self.a_p  + self.b_p))
        self.eta1 = float(self.a_e1 / self.b_e1)
        self.eta2 = float(self.a_e2 / self.b_e2)

        # Initialise latents: only flag large returns as candidate jumps
        thresh   = priors['_diagnostics']['jump_threshold']
        self.J   = (np.abs(self.r) > thresh).copy()
        self.Y   = np.where(self.J, self.r - self.mu * self.delta, 0.0)

        self._eta1_rej = 0
        self._eta1_tot = 0

    # -------------------------------------------------------------------------
    # Log-scale helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _log_phi(x: float) -> float:
        """log N(x; 0, 1) -- standard normal log-density"""
        return -0.5 * (np.log(2 * np.pi) + x * x)

    @staticmethod
    def _log_Phi(x: float) -> float:
        """log Phi(x) -- numerically stable standard normal log-CDF"""
        return stats.norm.logcdf(x)

    # -------------------------------------------------------------------------
    # Latent update: J_k and Y_k
    # -------------------------------------------------------------------------

    def _update_latent_k(self, k: int) -> float:
        """
        Collapsed Gibbs update for (J_k, Y_k) at observation k.
        """
        m_k   = self.r[k] - self.mu * self.delta
        s2    = self.sig2 * self.delta
        s     = np.sqrt(s2)
        y_min = self.y_min                        

        # ---- Branch weights (integrating out Y analytically) ----------------
        # Positive jump branch: ∫_{y_min}^∞ N(m_k; y, s²) p·η1·e^{-η1·y} dy
        # Completing the square gives:
        #   p · η1 · e^{-η1(m_k - y_min) + 0.5η1²s²} · Φ((m_k - η1s² - y_min)/s)
        arg_pos = (m_k - self.eta1 * s2 - y_min) / s   

        log_w_pos = (np.log(self.p)
                     + np.log(self.eta1)
                     - self.eta1 * (m_k - y_min)        
                     + 0.5 * self.eta1**2 * s2
                     + self._log_Phi(arg_pos))

        # Negative jump branch: ∫_{-∞}^{-y_min} N(m_k; y, s²) (1-p)·η2·e^{η2·y} dy
        arg_neg = (-m_k - self.eta2 * s2 - y_min) / s  # FIX 2  (was without y_min)

        log_w_neg = (np.log(1.0 - self.p)
                     + np.log(self.eta2)
                     + self.eta2 * (m_k + y_min)        
                     + 0.5 * self.eta2**2 * s2
                     + self._log_Phi(arg_neg))

        # ---- Marginal likelihoods -------------------------------------------
        # No-jump: log N(m_k; 0, s²)  -- exact, no change needed
        log_ell0 = self._log_phi(m_k / s) - 0.5 * np.log(s2)

        # Jump: logsumexp of branch weights -- 
        log_ell1 = logsumexp([log_w_pos, log_w_neg])

        # ---- Sample J_k  (Bernoulli with log-odds) -------------------------
        log_odds  = (np.log(self.pi)       + log_ell1) \
                  - (np.log(1.0 - self.pi) + log_ell0)
        p_jump    = 1.0 / (1.0 + np.exp(-log_odds))
        self.J[k] = (np.random.random() < p_jump)

        # ---- Sample Y_k (only if jump occurred) ----------------------------
        if self.J[k]:
            logit_q = log_w_pos - log_w_neg
            q       = 1.0 / (1.0 + np.exp(-logit_q))

            if np.random.random() < q:
                # Positive: TN(m_k - η1·s², s) on [y_min, ∞)  
                self.Y[k] = _sample_trunc_normal(
                    m_k - self.eta1 * s2, s, lower=y_min   # was lower=0.0
                )
            else:
                # Negative: TN(m_k + η2·s², s) on (−∞, −y_min]  
                self.Y[k] = _sample_trunc_normal(
                    m_k + self.eta2 * s2, s, upper=-y_min  # was upper=0.0
                )
        else:
            self.Y[k] = 0.0

        return log_ell0

    # -------------------------------------------------------------------------
    # Parameter full conditionals
    # -------------------------------------------------------------------------

    def _update_mu(self):
        """mu | rest ~ Normal  (Eq. 4.2)"""
        x    = self.r - self.J * self.Y
        v_mu = 1.0 / (1.0 / self.v0 + self.T * self.delta / self.sig2)
        m_mu = v_mu * (self.m0 / self.v0 + np.sum(x) / self.sig2)
        self.mu = np.random.normal(m_mu, np.sqrt(v_mu))

    def _update_sigma2(self):
        """sigma2 | rest ~ IG  (Eq. 4.4)"""
        resid  = self.r - self.mu * self.delta - self.J * self.Y
        a_post = self.a_s + 0.5 * self.T
        b_post = self.b_s + 0.5 * np.sum(resid**2) / self.delta
        self.sig2 = 1.0 / np.random.gamma(a_post, 1.0 / b_post)

    def _update_pi(self):
        """pi | rest ~ Beta  (Eq. 4.5)"""
        S = int(self.J.sum())
        self.pi = np.random.beta(self.a_pi + S, self.b_pi + self.T - S)

    def _update_p(self):
        """p | rest ~ Beta  (Eq. 4.6)"""
        n_pos = int(np.sum(self.J & (self.Y >= 0.0)))
        n_neg = int(np.sum(self.J & (self.Y <  0.0)))
        self.p = np.random.beta(self.a_p + n_pos, self.b_p + n_neg)

    def _update_eta1(self):
        """eta1 | rest ~ Gamma[truncated to (1,∞)]  (Eq. 4.7)"""
        mask   = self.J & (self.Y >= 0.0)
        n_pos  = int(mask.sum())
        sum_y  = float(self.Y[mask].sum())
        a_post = self.a_e1 + n_pos
        b_post = self.b_e1 + sum_y

        if TRUNC_ETA1:
            self._eta1_tot += 1
            for _ in range(10_000):
                candidate = np.random.gamma(a_post, 1.0 / b_post)
                if candidate > 1.0:
                    self.eta1 = candidate
                    break
                self._eta1_rej += 1
        else:
            self.eta1 = np.random.gamma(a_post, 1.0 / b_post)

    def _update_eta2(self):
        """eta2 | rest ~ Gamma  (Eq. 4.8)"""
        mask   = self.J & (self.Y < 0.0)
        n_neg  = int(mask.sum())
        sum_ay = float(np.abs(self.Y[mask]).sum())
        a_post = self.a_e2 + n_neg
        b_post = self.b_e2 + sum_ay
        self.eta2 = np.random.gamma(a_post, 1.0 / b_post)

    # -------------------------------------------------------------------------
    # Complete-data log-likelihood (convergence monitoring)
    # -------------------------------------------------------------------------

    def _log_likelihood(self) -> float:
        resid   = self.r - self.mu * self.delta - self.J * self.Y
        ll_diff = stats.norm.logpdf(resid, 0.0, np.sqrt(self.sig2 * self.delta)).sum()
        ll_J    = stats.bernoulli.logpmf(self.J.astype(int), self.pi).sum()

        jump_idx = np.where(self.J)[0]
        ll_Y     = 0.0
        for k in jump_idx:
            if self.Y[k] >= 0.0:
                ll_Y += np.log(self.p)       + np.log(self.eta1) - self.eta1 * self.Y[k]
            else:
                ll_Y += np.log(1.0 - self.p) + np.log(self.eta2) + self.eta2 * self.Y[k]

        return ll_diff + ll_J + ll_Y

    # -------------------------------------------------------------------------
    # Main sampling loop
    # -------------------------------------------------------------------------

    def run(self) -> dict:
        print()
        print("=" * 60)
        print("  Gibbs sampler -- Kou double-exponential jump-diffusion")
        print("  [All 4 identifiability fixes active]")
        print("=" * 60)
        print(f"  T={self.T}  N_iter={N_ITER}  burnin={N_BURNIN}  thin={N_THIN}")
        print(f"  Keeping {N_KEEP} draws  |  eta1>1 truncation: {TRUNC_ETA1}")
        print(f"  y_min={self.y_min:.6f}  |  pi prior: Beta({PI_PRIOR_A},{PI_PRIOR_B})")
        print("=" * 60)

        pb     = _ProgressBar(N_ITER)
        keep_i = 0

        for it in range(N_ITER):
            for k in range(self.T):
                self._update_latent_k(k)

            self._update_mu()
            self._update_sigma2()
            self._update_pi()
            self._update_p()
            self._update_eta1()
            self._update_eta2()

            if it >= N_BURNIN and (it - N_BURNIN) % N_THIN == 0:
                self.mu_s[keep_i]     = self.mu
                self.sig2_s[keep_i]   = self.sig2
                self.pi_s[keep_i]     = self.pi
                self.p_s[keep_i]      = self.p
                self.eta1_s[keep_i]   = self.eta1
                self.eta2_s[keep_i]   = self.eta2
                self.J_s[keep_i]      = self.J.copy()
                self.Y_s[keep_i]      = self.Y.copy()
                self.loglik_s[keep_i] = self._log_likelihood()
                keep_i += 1

            pb.update(it)

        pb.close()

        if TRUNC_ETA1 and self._eta1_tot > 0:
            rej_rate = 100.0 * self._eta1_rej / (self._eta1_tot * 10_000
                                                   + self._eta1_rej)
            print(f"  eta1 truncation rejection rate: {rej_rate:.2f}%")

        return self._compile()

    # -------------------------------------------------------------------------
    # Results compilation
    # -------------------------------------------------------------------------

    def _compile(self) -> dict:
        params = dict(mu=self.mu_s, sigma2=self.sig2_s, pi=self.pi_s,
                      p=self.p_s,   eta1=self.eta1_s,   eta2=self.eta2_s)

        derived = dict(
            lambda_annual = self.pi_s   / DELTA,
            sigma_annual  = np.sqrt(self.sig2_s * 252),
            mu_annual     = self.mu_s   * 252,
            E_pos_jump    = 1.0 / self.eta1_s,
            E_neg_jump    = -1.0 / self.eta2_s,
            kappa         = (self.p_s * self.eta1_s / (self.eta1_s - 1)
                             + (1 - self.p_s) * self.eta2_s / (self.eta2_s + 1)
                             - 1.0),
        )

        return dict(
            params      = params,
            derived     = derived,
            latent      = dict(J=self.J_s, Y=self.Y_s),
            loglik      = self.loglik_s,
            eta1_rej    = self._eta1_rej / max(self._eta1_tot, 1),
            T           = self.T,
            delta       = DELTA,
            y_min       = self.y_min,       # stored for downstream use
            priors      = self.priors,
        )


# =============================================================================
# PART 2 -- MCMC DIAGNOSTICS
# =============================================================================

def _ess(chain: np.ndarray) -> float:
    """ESS via the initial monotone sequence estimator (Geyer 1992)."""
    n   = len(chain)
    x   = chain - chain.mean()
    f   = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(f * np.conj(f)))[:n] / (n * np.var(chain))
    ess_n = n
    for t in range(1, n // 2):
        gamma = acf[2*t - 1] + acf[2*t]
        if gamma <= 0:
            break
        ess_n = min(ess_n, n / (2 * t * gamma + 1 - 1/n))
    return max(float(ess_n), 1.0)


def _r_hat(chains: list) -> float:
    """Gelman-Rubin R-hat for a list of equal-length chains."""
    M = len(chains)
    N = len(chains[0])
    chain_means = np.array([c.mean() for c in chains])
    chain_vars  = np.array([c.var(ddof=1) for c in chains])
    grand_mean  = chain_means.mean()
    B           = N / (M - 1) * np.sum((chain_means - grand_mean)**2)
    W           = chain_vars.mean()
    var_plus    = ((N - 1) * W + B) / N
    return float(np.sqrt(var_plus / W)) if W > 0 else np.nan


def print_summary(results: dict):
    params  = results['params']
    derived = results['derived']
    y_min   = results.get('y_min', 0.0)

    SEP = "-" * 72
    print()
    print("=" * 72)
    print("  POSTERIOR SUMMARY -- Kou Jump-Diffusion ")
    print(f"  y_min={y_min:.6f}  |  pi prior: Beta({PI_PRIOR_A},{PI_PRIOR_B})")
    print("=" * 72)
    print(f"  {'Parameter':<22} {'Mean':>10} {'Std':>9} "
          f"{'2.5%':>10} {'50%':>10} {'97.5%':>10} {'ESS':>6} {'R-hat':>6}")
    print("  " + SEP)

    all_items = list(params.items()) + [
        ('lambda_annual', derived['lambda_annual']),
        ('sigma_annual',  derived['sigma_annual']),
        ('mu_annual',     derived['mu_annual']),
        ('E_pos_jump',    derived['E_pos_jump']),
        ('E_neg_jump',    derived['E_neg_jump']),
        ('kappa',         derived['kappa']),
    ]

    for name, smp in all_items:
        q25, q50, q975 = np.percentile(smp, [2.5, 50, 97.5])
        ess  = _ess(smp)
        half = len(smp) // 2
        rhat = _r_hat([smp[:half], smp[half:]])
        print(f"  {name:<22} {smp.mean():>10.5f} {smp.std():>9.5f} "
              f"{q25:>10.5f} {q50:>10.5f} {q975:>10.5f} "
              f"{ess:>6.0f} {rhat:>6.3f}")

    print()
    lam = results['params']['pi'].mean() / DELTA
    status = '[OK -- Bernoulli valid]' if lam <= 15 else '[WARN: raise PI_PRIOR_B or Y_MIN_SIGMA]'
    print(f"  Bernoulli validity:  E[lambda] = {lam:.1f} jumps/yr  {status}")
    print("=" * 72)
    print()


# =============================================================================
# PART 2 -- POSTERIOR VISUALISATION
# =============================================================================

def plot_all(results: dict, returns: np.ndarray, dates, out_dir: str):

    params  = results['params']
    derived = results['derived']
    J_post  = results['latent']['J']
    y_min   = results.get('y_min', 0.0)

    def _ci(smp, lo=2.5, hi=97.5):
        return np.percentile(smp, lo), np.percentile(smp, hi)

    # -- Figure 1: Trace plots ------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    param_labels = ['mu (daily)', 'sigma^2 (daily)', 'pi (jump prob)',
                    'p (P[jump>0])', 'eta1 (pos rate)', 'eta2 (neg rate)']
    for i, (name, lbl) in enumerate(zip(params, param_labels)):
        ax  = axes[i]
        smp = params[name]
        ax.plot(smp, lw=0.5, alpha=0.8, color='steelblue')
        ax.axhline(smp.mean(), color='crimson', ls='--', lw=1.2,
                   label=f'mean={smp.mean():.4f}')
        ax.set_title(f'Trace: {lbl}', fontweight='bold')
        ax.set_xlabel('Post-burnin draw (thinned)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.suptitle('Trace Plots -- Gibbs Sampler ',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_traces.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_traces.png")

    # -- Figure 2: Posterior densities ----------------------------------------
    plot_cfg = [
        ('mu',            params['mu'],             'blue',        'mu (daily drift)'),
        ('sigma_annual',  derived['sigma_annual'],  'teal',        'sigma (annual)'),
        ('lambda_annual', derived['lambda_annual'], 'darkorange',  'lambda (jumps/yr)'),
        ('pi',            params['pi'],             'purple',      'pi (daily jump prob)'),
        ('p',             params['p'],              'saddlebrown', 'p = P[jump > 0]'),
        ('eta1',          params['eta1'],           'firebrick',   'eta1 (pos decay rate)'),
        ('eta2',          params['eta2'],           'darkred',     'eta2 (neg decay rate)'),
        ('E_pos_jump',    derived['E_pos_jump'],    'green',       'E[Y+] = 1/eta1'),
        ('E_neg_jump',    derived['E_neg_jump'],    'darkgreen',   'E[Y-] = -1/eta2'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    for i, (_, smp, color, lbl) in enumerate(plot_cfg):
        ax = axes[i]
        ax.hist(smp, bins=50, density=True, alpha=0.65,
                color=color, edgecolor='white', lw=0.3)
        lo, hi = _ci(smp)
        ax.axvline(smp.mean(), color='black',   ls='--', lw=1.4,
                   label=f'mean={smp.mean():.4f}')
        ax.axvline(lo,         color='dimgray', ls=':',  lw=1.0, label='95% CI')
        ax.axvline(hi,         color='dimgray', ls=':',  lw=1.0)
        ax.set_title(lbl, fontweight='bold')
        ax.set_xlabel(lbl)
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    plt.suptitle('Posterior Marginal Densities', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_posteriors.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_posteriors.png")

    # -- Figure 3: ACF plots --------------------------------------------------
    def _acf(x, nlags=100):
        x = x - x.mean()
        n = len(x)
        full_acov = np.fft.irfft(np.abs(np.fft.rfft(x, n=2*n))**2)[:n]
        return (full_acov / full_acov[0])[:nlags+1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, (name, lbl) in enumerate(zip(params, param_labels)):
        ax       = axes[i]
        acf_vals = _acf(params[name])
        lags     = np.arange(len(acf_vals))
        ax.bar(lags, acf_vals, width=0.9, color='steelblue', alpha=0.7)
        ax.axhline(0,   color='k',   lw=0.5)
        ax.axhline( 1.96 / np.sqrt(N_KEEP), color='red', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(-1.96 / np.sqrt(N_KEEP), color='red', ls='--', lw=0.8, alpha=0.7)
        ax.set_title(f'ACF: {lbl}', fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylim(-0.3, 1.0)
        ax.grid(alpha=0.25)
    plt.suptitle('Autocorrelation Functions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_acf.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_acf.png")

    # -- Figure 4: Log-likelihood trace ---------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results['loglik'], lw=0.6, color='steelblue', alpha=0.8)
    ax.axhline(np.median(results['loglik']), color='crimson', ls='--', lw=1.2,
               label=f"median={np.median(results['loglik']):.1f}")
    ax.set_title('Complete-Data Log-Likelihood (post burn-in)', fontweight='bold')
    ax.set_xlabel('Post-burnin draw (thinned)')
    ax.set_ylabel('log p(r, J, Y | theta)')
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_loglik.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_loglik.png")

    # -- Figure 5: Jump detection ---------------------------------------------
    p_jump = J_post.mean(axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={'height_ratios': [2, 1]},
                                    sharex=True)
    ax1.plot(dates, returns, color='dimgray', lw=0.6, alpha=0.7, label='log return')
    ax1.fill_between(dates, returns.min(), returns.max(),
                     where=(p_jump > 0.5), color='salmon', alpha=0.25,
                     label='P(jump) > 0.5')
    ax1.axhline(0, color='k', lw=0.3)
    ax1.set_ylabel('Log Return')
    ax1.set_title('Returns with Posterior Jump Probability', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    ax2.bar(dates, p_jump, width=1.5, color='crimson', alpha=0.6, edgecolor='none')
    ax2.axhline(0.5, color='black', ls='--', lw=0.8, alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('P(J_k = 1)')
    ax2.set_xlabel('Date')
    ax2.grid(alpha=0.2)
    n_high = int((p_jump > 0.5).sum())
    ax2.set_title(f'Posterior Jump Probabilities  ({n_high} days with P > 0.5)',
                  fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_jumps.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_jumps.png")

    # -- Figure 6: Jump sizes + prior/posterior pi ----------------------------
    # FIX 2: sample from shifted exponential (memoryless property)
    idx_pred = np.random.choice(N_KEEP, 2000, replace=True)
    y_pred   = []
    for i in idx_pred:
        if np.random.random() < params['p'][i]:
            y_pred.append(y_min + np.random.exponential(1.0 / params['eta1'][i]))
        else:
            y_pred.append(-(y_min + np.random.exponential(1.0 / params['eta2'][i])))
    y_pred = np.array(y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(y_pred, bins=60, density=True, alpha=0.7, color='steelblue',
                 edgecolor='white', lw=0.3)
    axes[0].axvline(0, color='k', lw=0.8)
    axes[0].set_title('Posterior Predictive Jump Sizes', fontweight='bold')
    axes[0].set_xlabel('Jump size Y_k (log-return)')
    axes[0].set_ylabel('Density')
    axes[0].grid(alpha=0.25)

    pos = y_pred[y_pred > 0]
    neg = np.abs(y_pred[y_pred < 0])
    if len(pos):
        axes[1].hist(pos, bins=40, density=True, alpha=0.6, color='green',
                     edgecolor='white', lw=0.3, label='positive')
        x_pos = np.linspace(y_min, pos.max(), 200)
        eta1_m = np.mean(params['eta1'])
        axes[1].plot(x_pos, eta1_m * np.exp(-eta1_m * (x_pos - y_min)),
                     'k--', lw=1.5, label='Exp(E[eta1]) shifted')
    if len(neg):
        axes[1].hist(neg, bins=40, density=True, alpha=0.6, color='red',
                     edgecolor='white', lw=0.3, label='|negative|')
        x_neg = np.linspace(y_min, neg.max(), 200)
        eta2_m = np.mean(params['eta2'])
        axes[1].plot(x_neg, eta2_m * np.exp(-eta2_m * (x_neg - y_min)),
                     'gray', ls='--', lw=1.5, label='Exp(E[eta2]) shifted')
    axes[1].set_title('Jump Tails  (y_min-shifted Exp fit)', fontweight='bold')
    axes[1].set_xlabel('Magnitude')
    axes[1].set_ylabel('Density')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    a_pi0, b_pi0 = results['priors']['pi']
    x_pi         = np.linspace(0, 0.06, 400)
    axes[2].hist(params['pi'], bins=50, density=True, alpha=0.6, color='purple',
                 edgecolor='white', lw=0.3, label='Posterior')
    axes[2].plot(x_pi, stats.beta.pdf(x_pi, a_pi0, b_pi0),
                 'k--', lw=1.5, label=f'Prior Beta({a_pi0},{b_pi0})')
    axes[2].set_title('Prior vs Posterior: pi  [FIX 3]', fontweight='bold')
    axes[2].set_xlabel('pi')
    axes[2].set_ylabel('Density')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    plt.suptitle('Jump Sizes & Prior/Posterior Comparison',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_jump_sizes.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_jump_sizes.png")

    # -- Figure 7: Prior vs Posterior  ----------------------
    prior_dists = {
        'mu':     ('norm',  results['priors']['mu']),
        'sigma2': ('ig',    results['priors']['sigma2']),
        'pi':     ('beta',  results['priors']['pi']),
        'p':      ('beta',  results['priors']['p']),
        'eta1':   ('gamma', results['priors']['eta1']),
        'eta2':   ('gamma', results['priors']['eta2']),
    }
    pv_labels = ['mu (daily)', 'sigma^2 (daily)', 'pi  [FIX3]',
                 'p', 'eta1  [FIX4]', 'eta2  [FIX4]']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (name, lbl) in enumerate(zip(prior_dists, pv_labels)):
        ax             = axes[i]
        smp            = params[name]
        dtype, hparams = prior_dists[name]

        lo_x  = np.percentile(smp, 0.5)
        hi_x  = np.percentile(smp, 99.5)
        x_rng = np.linspace(lo_x, hi_x, 400)

        if dtype == 'norm':
            m0v, v0v  = hparams
            prior_pdf = stats.norm.pdf(x_rng, m0v, np.sqrt(v0v))
        elif dtype == 'ig':
            a, b      = hparams
            prior_pdf = stats.invgamma.pdf(x_rng, a, scale=b)
        elif dtype == 'beta':
            a, b      = hparams
            prior_pdf = stats.beta.pdf(x_rng, a, b)
        elif dtype == 'gamma':
            a, b      = hparams
            prior_pdf = stats.gamma.pdf(x_rng, a, scale=1.0/b)

        ax.hist(smp, bins=50, density=True, alpha=0.55,
                color='steelblue', edgecolor='white', lw=0.3, label='Posterior')
        ax.plot(x_rng, prior_pdf, 'crimson', lw=2.0, label='Prior')
        ax.set_title(lbl, fontweight='bold')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.suptitle('Prior (red) vs Posterior (blue) -- All Parameters',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_prior_vs_posterior.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_prior_vs_posterior.png")


# =============================================================================
# PART 2 -- POSTERIOR PREDICTIVE CHECK
# =============================================================================

def plot_predictive_check(results: dict, returns: np.ndarray, out_dir: str):
    """
    We should here note jump sizes drawn via Y ~ y_min + Exp(eta) (memoryless shift),
    which is exact simulation from a Exp(eta) truncated to [y_min, inf).
    """
    params = results['params']
    T      = results['T']
    delta  = results['delta']
    y_min  = results.get('y_min', 0.0)

    n_pred = min(500, N_KEEP)
    idx    = np.random.choice(N_KEEP, n_pred, replace=False)
    sims   = []
    for i in idx:
        mu_i   = params['mu'][i]
        sig2_i = params['sigma2'][i]
        pi_i   = params['pi'][i]
        p_i    = params['p'][i]
        e1_i   = params['eta1'][i]
        e2_i   = params['eta2'][i]

        eps   = np.random.normal(0, np.sqrt(sig2_i * delta), T)
        J     = np.random.random(T) < pi_i
        Y     = np.zeros(T)
        j_idx = np.where(J)[0]

        if len(j_idx):
            signs   = np.random.random(len(j_idx)) < p_i
            pos_idx = j_idx[signs]
            neg_idx = j_idx[~signs]
            if len(pos_idx):
                # Exact truncated Exp via memoryless shift (FIX 2)
                Y[pos_idx] =  y_min + np.random.exponential(1.0 / e1_i, len(pos_idx))
            if len(neg_idx):
                Y[neg_idx] = -(y_min + np.random.exponential(1.0 / e2_i, len(neg_idx)))

        sims.append(mu_i * delta + eps + Y)

    sim_flat = np.concatenate(sims)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(returns,  bins=80, density=True, alpha=0.5, color='steelblue',
                 label='Observed', edgecolor='white', lw=0.3)
    axes[0].hist(sim_flat, bins=80, density=True, alpha=0.35, color='tomato',
                 label='Posterior predictive', edgecolor='white', lw=0.3)
    axes[0].set_title('Return Distribution: Observed vs Simulated', fontweight='bold')
    axes[0].set_xlabel('Log Return')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    q_obs  = np.percentile(returns,  np.linspace(1, 99, 200))
    q_pred = np.percentile(sim_flat, np.linspace(1, 99, 200))
    axes[1].scatter(q_obs, q_pred, s=10, alpha=0.6, color='steelblue')
    mn = min(q_obs.min(), q_pred.min())
    mx = max(q_obs.max(), q_pred.max())
    axes[1].plot([mn, mx], [mn, mx], 'crimson', lw=1.5, label='y=x')
    axes[1].set_title('QQ: Observed vs Posterior Predictive', fontweight='bold')
    axes[1].set_xlabel('Observed quantiles')
    axes[1].set_ylabel('Simulated quantiles')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    obs_kurt = stats.kurtosis(returns, fisher=True)
    sim_kurt = np.array([stats.kurtosis(s, fisher=True) for s in sims])
    axes[2].hist(sim_kurt, bins=40, density=True, alpha=0.6, color='teal',
                 label='Simulated excess kurtosis')
    axes[2].axvline(obs_kurt, color='crimson', lw=2,
                    label=f'Observed ({obs_kurt:.2f})')
    axes[2].set_title('Excess Kurtosis: Posterior Predictive Check', fontweight='bold')
    axes[2].set_xlabel('Excess Kurtosis')
    axes[2].set_ylabel('Density')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    plt.suptitle('Posterior Predictive Check', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'levy_ppc.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] levy_ppc.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = os.path.dirname(os.path.abspath(EXCEL_PATH))

    # ---- Part 1: data + priors ----------------------------------------------
    df = fetch_data(EXCEL_PATH)
    df = df.loc[DATE_START:DATE_END]
    if df.empty:
        raise ValueError(f"No data between {DATE_START} and {DATE_END}.")
    print(f"[FILTER] Using {len(df)} trading days  ({DATE_START} to {DATE_END})")

    returns = np.log(df['adjusted_close'] / df['adjusted_close'].shift(1)).dropna()
    dates   = returns.index

    priors = compute_priors(returns)
    print_priors(priors)
    plot_diagnostics(df, returns, priors)

    if SAVE_PRIORS:
        pkl_path = os.path.join(out_dir, 'sony_empirical_priors.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(priors, f)
        print(f"[SAVED] Priors -> {pkl_path}")

    # ---- Part 2: Gibbs sampler ----------------------------------------------
    ret_arr = returns.values
    print(f"\n[OK] {len(ret_arr)} returns  "
          f"({str(dates[0])[:10]} to {str(dates[-1])[:10]})")

    sampler = KouGibbsSampler(ret_arr, priors)
    results = sampler.run()

    print_summary(results)

    if PLOT_RESULTS:
        plot_all(results, ret_arr, dates, out_dir)
        plot_predictive_check(results, ret_arr, out_dir)

    if SAVE_RESULTS:
        pkl_path = os.path.join(out_dir, 'sony_mcmc_results_2025.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"[SAVED] Full results -> {pkl_path}")

        rows = []
        for name, smp in (list(results['params'].items())
                          + list(results['derived'].items())):
            q25, q50, q975 = np.percentile(smp, [2.5, 50, 97.5])
            rows.append(dict(parameter=name,
                             mean=smp.mean(), std=smp.std(),
                             q025=q25, median=q50, q975=q975,
                             ess=_ess(smp)))
        csv_path = os.path.join(out_dir, 'sony_posterior_summary_2025.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')
        print(f"[SAVED] Summary CSV -> {csv_path}")

    return df, priors, results


if __name__ == '__main__':
    df, priors, results = main()