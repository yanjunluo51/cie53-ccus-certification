"""
newfamily_geoasian_M4.py — NEW FAMILY (review issue M4)
=======================================================
Family A: Bermudan GEOMETRIC-AVERAGE ASIAN CALL on a single GBM.

WHY THIS IS A GENUINELY NEW FAMILY
----------------------------------
  * NOT CCUS, NOT the 5D max-call, NOT the 1D Longstaff-Schwartz vanilla put.
  * Path-dependent payoff: the exercise value at t depends on the *running
    geometric average* G_t = (prod_{j=0}^{t} S_j)^{1/(t+1)}, NOT on S_t alone.
    The effective state is 2-D (S_t, G_t) and the continuation-value regression
    is over (log S, log G), a structurally different basis from the CCUS
    (P,C,R) basis and the max-call (S_1..S_d) basis.
  * Payoff:  Z_t = max(G_t - K, 0).

VALIDATION GATE (mandatory, independent reference)
--------------------------------------------------
For the GEOMETRIC average the running average G_T is itself lognormal, so the
*European* discrete geometric-average Asian call has an EXACT closed form
(Kemna-Vorst, discrete-monitoring version). We use this two ways:
  (1) Monte-Carlo the European payoff with the SAME simulator/payoff code and
      check it reproduces the closed form to <~1%  ->  validates the engine.
  (2) The Bermudan value must satisfy  V_Bermudan >= V_European_geo  (strictly
      more exercise opportunity) and lie modestly above it. The LSM lower bound
      is checked against this hard inequality.
A hybrid certificate UB then sandwiches the true Bermudan price from above and
must satisfy E[UB] >= V_LB (dual feasibility).

rho_res diagnostic (paper's definition, exp2_4_residual_dominance.py)
---------------------------------------------------------------------
    rho_res = sum_t mean|T2_t| / ( sum_t mean|T1_t| + sum_t mean|T2_t| )
over non-terminal steps, where
    T1_t = E[ C_hat_t(S_t,G_t) | F_{t-1} ]                  (closed-form)
    T2_t = E[ (Z_t - C_hat_t)_+ | F_{t-1} ]                 (residual MC)
Because (S_t, G_t) given F_{t-1} is jointly lognormal in closed form, T1_t is
computed analytically (Gaussian raw moments of standardized log-coordinates),
exactly mirroring closed_form_E_basis_nd in maxcall_5d_benchmark.py.

We sweep volatility sigma to scan rho_res across the transition band
[0.10, 0.30] and test whether the hybrid gap rises with rho_res WITHIN family.

Seeds reported; SEs reported. REAL CODE; no fabricated numbers.
"""
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from math import comb, log, sqrt, exp, erf
from scipy.special import ndtri

np.seterr(over="raise", divide="raise", invalid="raise")

# ============================================================
#  Fixed contract parameters
# ============================================================
S0      = 100.0
K       = 100.0
R       = 0.05
DELTA   = 0.0            # no dividend
T_YEARS = 1.0
N_STEPS = 10            # 10 monitoring/exercise dates (incl. averaging points)
DT      = T_YEARS / N_STEPS
DEGREE  = 3
N_TRAIN = int(os.environ.get("GEO_NTRAIN", "40000"))
N_OUT   = int(os.environ.get("GEO_NOUT", "40000"))
N_RES   = int(os.environ.get("GEO_NRES", "50"))
N_IN    = int(os.environ.get("GEO_NIN", "100"))
N_SEEDS = int(os.environ.get("GEO_SEEDS", "5"))
N_SELECT = int(os.environ.get("GEO_NSELECT", "10000"))
WORKERS = max(1, int(os.environ.get("GEO_WORKERS", "2")))


# ============================================================
#  Simulation of (S_t, running geometric average G_t)
# ============================================================
def simulate_paths(n_paths, sigma, rng):
    """Return S (n,T+1) and G (n,T+1) where G_t = (prod_{j=0}^t S_j)^{1/(t+1)}.

    Averaging starts at t=0 (includes S0). Exercise allowed at t=1..T.
    """
    drift = (R - DELTA - 0.5 * sigma**2) * DT
    vol = sigma * sqrt(DT)
    S = np.empty((n_paths, N_STEPS + 1))
    S[:, 0] = S0
    for t in range(1, N_STEPS + 1):
        Z = rng.standard_normal(n_paths)
        S[:, t] = S[:, t - 1] * np.exp(drift + vol * Z)
    logS = np.log(S)
    cum_logS = np.cumsum(logS, axis=1)
    counts = np.arange(1, N_STEPS + 2)  # t+1 terms at column t
    G = np.exp(cum_logS / counts[None, :])
    return S, G


def payoff(G_t):
    return np.maximum(G_t - K, 0.0)


# ============================================================
#  VALIDATION: exact discrete-geometric European Asian closed form
# ============================================================
def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def european_geo_asian_closed_form(sigma):
    """Exact price of a European call on the discrete geometric average
    G_T = (prod_{j=0}^{N} S_{t_j})^{1/(N+1)} with monitoring at t_j = j*DT,
    j=0..N (N=N_STEPS). Standard discrete Kemna-Vorst.

    log G_T ~ Normal(mu_G, var_G) under Q with:
       log S_{t_j} = log S0 + (r-delta-0.5 sigma^2) t_j + sigma W_{t_j}.
    Average over j of the deterministic part -> mu_drift.
    Var of (1/(N+1)) sum_j W_{t_j}: cov(W_{t_i},W_{t_j}) = min(t_i,t_j).
    """
    n = N_STEPS + 1                     # number of averaging points j=0..N
    times = np.arange(n) * DT           # t_0=0 ... t_N=T
    a = (R - DELTA - 0.5 * sigma**2)
    mu_drift = log(S0) + a * times.mean()
    # variance of (sigma/n) sum_j W_{t_j}
    tmin = np.minimum.outer(times, times)
    var_G = (sigma**2 / n**2) * tmin.sum()
    sig_G = sqrt(var_G)
    # E[G_T] = exp(mu_drift + 0.5 var_G)
    d1 = (mu_drift + var_G - log(K)) / sig_G
    d2 = d1 - sig_G
    price = exp(-R * T_YEARS) * (
        exp(mu_drift + 0.5 * var_G) * _norm_cdf(d1) - K * _norm_cdf(d2)
    )
    return price


# ============================================================
#  Standardized-log basis on (log S, log G)
# ============================================================
def compute_std_stats(S_tr, G_tr):
    lS = np.log(np.maximum(S_tr[:, 1:].reshape(-1), 1e-12))
    lG = np.log(np.maximum(G_tr[:, 1:].reshape(-1), 1e-12))
    return (lS.mean(), max(lS.std(), 1e-8), lG.mean(), max(lG.std(), 1e-8))


def _multi_indices(degree):
    idx = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            idx.append((i, j))
    return idx


INDICES = _multi_indices(DEGREE)


def basis(S, G, stats):
    muS, sdS, muG, sdG = stats
    zS = (np.log(np.maximum(S, 1e-12)) - muS) / sdS
    zG = (np.log(np.maximum(G, 1e-12)) - muG) / sdG
    feats = np.ones((S.shape[0], len(INDICES)))
    for col, (i, j) in enumerate(INDICES):
        if i > 0:
            feats[:, col] *= zS ** i
        if j > 0:
            feats[:, col] *= zG ** j
    return feats


# ============================================================
#  Closed-form T1 = E[ C_hat(S_t,G_t) | F_{t-1} ]
# ============================================================
def gaussian_raw_moment(mu, var, k):
    if k == 0:
        return np.ones_like(mu)
    acc = np.zeros_like(mu, dtype=np.float64)
    for j in range(k // 2 + 1):
        c = comb(k, 2 * j)
        df = 1.0
        for i in range(2 * j - 1, 0, -2):
            df *= i
        acc += c * df * (var ** j) * (mu ** (k - 2 * j))
    return acc


def gaussian_cross_moment(muX, muY, varX, varY, covXY, kx, ky):
    """E[X^kx Y^ky] for bivariate normal, via conditioning Y|X.
    Y|X=x ~ N(muY + b(x-muX), varY - b^2 varX), b = covXY/varX.
    E[X^kx Y^ky] = E_X[ X^kx * E[Y^ky | X] ]; expand the conditional moment
    using Hermite-style raw moments. Vectorized over the path dimension.
    """
    if ky == 0:
        return gaussian_raw_moment(muX, varX, kx)
    if kx == 0:
        return gaussian_raw_moment(muY, varY, ky)
    b = covXY / varX
    cond_var = varY - b * b * varX
    cond_var = np.maximum(cond_var, 0.0)
    # E[Y^ky | X=x] with mean m(x)=muY+b(x-muX): use raw moments in (m,cond_var).
    # We need E_X[ X^kx * (sum over decomposition) ]. Expand m(x)^... via binomial
    # within gaussian_raw_moment by treating m as a vector; but m depends on x
    # which is the integration variable. So Monte-Carlo-free exact expansion:
    # E[X^kx Y^ky] = sum_{p} C(ky,2p)(2p-1)!! cond_var^p E[X^kx m(X)^{ky-2p}].
    acc = np.zeros_like(muX, dtype=np.float64)
    for p in range(ky // 2 + 1):
        c = comb(ky, 2 * p)
        df = 1.0
        for i in range(2 * p - 1, 0, -2):
            df *= i
        q = ky - 2 * p
        # E[ X^kx (muY - b muX + b X)^q ] ; let A = muY - b muX
        A = muY - b * muX
        inner = np.zeros_like(muX, dtype=np.float64)
        for m in range(q + 1):
            cm = comb(q, m)
            inner += cm * (A ** (q - m)) * (b ** m) * gaussian_raw_moment(muX, varX, kx + m)
        acc += c * df * (cond_var ** p) * inner
    return acc


def closed_form_T1(S_prev, G_prev, t_prev, beta, stats, sigma):
    """E[ sum_col beta_col * zS^i zG^j | F_{t-1} ].
    Given S_{t-1}, G_{t-1} (geom avg over t_prev+1 points j=0..t_prev),
       log S_t = log S_{t-1} + a*DT + sigma*sqrt(DT) eps
       log G_t = (t_prev+1)/(t_prev+2) * log G_{t-1} + 1/(t_prev+2) * log S_t
    Both linear in the single shock eps ~ N(0,1)  =>  jointly normal.
    """
    muS, sdS, muG, sdG = stats
    a = (R - DELTA - 0.5 * sigma**2)
    w = (t_prev + 1) / (t_prev + 2)      # weight on old average
    v = 1.0 / (t_prev + 2)               # weight on new log S_t

    mean_logSt = np.log(np.maximum(S_prev, 1e-12)) + a * DT
    sd_logSt = sigma * sqrt(DT)
    mean_logGt = w * np.log(np.maximum(G_prev, 1e-12)) + v * mean_logSt
    sd_logGt = v * sd_logSt
    cov = v * (sd_logSt ** 2)            # cov(logS_t, logG_t)

    # standardized coords
    muZS = (mean_logSt - muS) / sdS
    varZS = (sd_logSt / sdS) ** 2
    muZG = (mean_logGt - muG) / sdG
    varZG = (sd_logGt / sdG) ** 2
    covZ = cov / (sdS * sdG)

    out = np.zeros_like(S_prev, dtype=np.float64)
    for col, (i, j) in enumerate(INDICES):
        if beta[col] == 0.0:
            continue
        m = gaussian_cross_moment(muZS, muZG, varZS, varZG, covZ, i, j)
        out += beta[col] * m
    return out


# ============================================================
#  One-step inner simulator from (S_prev, G_prev) at step t
# ============================================================
def inner_step(S_prev, G_prev, t_prev, n_res, sigma, rng):
    """Return S_t, G_t arrays of shape (n_outer, n_res)."""
    a = (R - DELTA - 0.5 * sigma**2)
    n = S_prev.shape[0]
    Z = rng.standard_normal((n, n_res))
    logSt = np.log(np.maximum(S_prev, 1e-12))[:, None] + a * DT + sigma * sqrt(DT) * Z
    St = np.exp(logSt)
    w = (t_prev + 1) / (t_prev + 2)
    v = 1.0 / (t_prev + 2)
    logGt = w * np.log(np.maximum(G_prev, 1e-12))[:, None] + v * logSt
    Gt = np.exp(logGt)
    return St, Gt


def inner_step_stratified(S_prev, G_prev, t_prev, n_res, sigma, rng):
    """Randomized Latin-hypercube one-step transitions, independently by row."""
    a = (R - DELTA - 0.5 * sigma**2)
    n = S_prev.shape[0]
    n_res = int(n_res)
    if n_res < 1:
        raise ValueError("n_res must be positive")
    order = np.argsort(rng.random((n, n_res)), axis=1)
    uniforms = (order + rng.random((n, n_res))) / n_res
    eps = np.finfo(float).eps
    shocks = ndtri(np.clip(uniforms, eps, 1.0 - eps))
    logSt = (
        np.log(np.maximum(S_prev, 1e-12))[:, None]
        + a * DT
        + sigma * sqrt(DT) * shocks
    )
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        St = np.exp(logSt)
        w = (t_prev + 1) / (t_prev + 2)
        v = 1.0 / (t_prev + 2)
        logGt = (
            w * np.log(np.maximum(G_prev, 1e-12))[:, None]
            + v * logSt
        )
        Gt = np.exp(logGt)
    if not np.isfinite(St).all() or not np.isfinite(Gt).all():
        raise FloatingPointError("non-finite stratified geometric-Asian state")
    return St, Gt


# ============================================================
#  LSM training
# ============================================================
def lsm_train(S, G, stats):
    n = S.shape[0]
    T = N_STEPS
    disc = exp(-R * DT)
    V = payoff(G[:, T])
    betas = [None] * (T)
    for t in range(T - 1, 0, -1):
        ex = payoff(G[:, t])
        itm = ex > 0
        if itm.sum() < len(INDICES) + 5:
            V = disc * V
            continue
        Y = disc * V[itm]
        Phi = basis(S[itm, t], G[itm, t], stats)
        beta = np.linalg.lstsq(Phi, Y, rcond=None)[0]
        betas[t] = beta
        C_hat = np.einsum("ij,j->i", Phi, beta)
        ex_now = ex[itm] > C_hat
        idx = np.where(itm)[0]
        V_new = disc * V.copy()
        V_new[idx[ex_now]] = ex[itm][ex_now]
        V = V_new
    return betas


def lsm_lower_bound(S, G, betas, stats, return_stats=False):
    n = S.shape[0]
    T = N_STEPS
    disc = exp(-R * DT)
    V = payoff(G[:, T])
    for t in range(T - 1, 0, -1):
        ex = payoff(G[:, t])
        if betas[t] is not None:
            C_hat = np.einsum("ij,j->i", basis(S[:, t], G[:, t], stats), betas[t])
            ex_now = ex > C_hat
        else:
            ex_now = np.zeros(n, dtype=bool)
        V = np.where(ex_now, ex, disc * V)
    # Discounting fix: the backward loop stops at t=1, so this value is priced at time 1
    # and needs one more discount step to be a time-0 price. The upper bound is discounted
    # to time 0 with exp(-R*arange(T+1)*DT). Mixing the two would shrink the gap systematically.
    V = disc * V
    mean = V.mean()
    if return_stats:
        return mean, V.std(ddof=1) / sqrt(n), V
    return mean


# ============================================================
#  Residual-nested hybrid AB (conservative) + rho_res
# ============================================================
def hybrid_and_rho(
    S_out, G_out, betas, stats, sigma, n_res, rng,
    return_rho_stats=False, residual_mode="stratified",
):
    if residual_mode not in {"stratified", "plain"}:
        raise ValueError("residual_mode must be 'stratified' or 'plain'")
    step = inner_step_stratified if residual_mode == "stratified" else inner_step
    n = S_out.shape[0]
    T = N_STEPS
    disc_f = np.exp(-R * np.arange(T + 1) * DT)

    Z_all = np.zeros((n, T + 1))
    Q = np.zeros((n, T + 1))
    for t in range(T + 1):
        Z_all[:, t] = payoff(G_out[:, t])
    for t in range(1, T + 1):
        if t < T and betas[t] is not None:
            C_hat = np.einsum(
                "ij,j->i", basis(S_out[:, t], G_out[:, t], stats), betas[t])
        else:
            C_hat = np.zeros(n)
        Q[:, t] = np.maximum(Z_all[:, t], C_hat)
    Q[:, 0] = Z_all[:, 0]

    M = np.zeros((n, T + 1))
    t1_abs_sum = 0.0
    t2_abs_sum = 0.0
    t1_abs_by_path = np.zeros(n)
    t2_abs_by_path = np.zeros(n)
    n_counted = 0
    for t in range(1, T + 1):
        if t == T:
            St, Gt = step(
                S_out[:, t - 1], G_out[:, t - 1], t - 1,
                n_res, sigma, rng)
            E_Q = payoff(Gt).mean(axis=1)
        else:
            beta = betas[t] if betas[t] is not None else np.zeros(len(INDICES))
            T1 = closed_form_T1(S_out[:, t - 1], G_out[:, t - 1], t - 1, beta, stats, sigma)
            St, Gt = step(
                S_out[:, t - 1], G_out[:, t - 1], t - 1,
                n_res, sigma, rng)
            Z_in = payoff(Gt)
            C_in = np.einsum(
                "ij,j->i", basis(St.reshape(-1), Gt.reshape(-1), stats), beta
            ).reshape(St.shape)
            T2 = np.maximum(Z_in - C_in, 0.0).mean(axis=1)
            E_Q = T1 + T2
            t1_abs_sum += np.abs(T1).mean()
            t2_abs_sum += np.abs(T2).mean()
            t1_abs_by_path += np.abs(T1)
            t2_abs_by_path += np.abs(T2)
            n_counted += 1
        dM = disc_f[t] * Q[:, t] - disc_f[t] * E_Q
        M[:, t] = M[:, t - 1] + dM

    pathwise = np.full(n, -np.inf)
    for t in range(T + 1):
        pathwise = np.maximum(pathwise, disc_f[t] * Z_all[:, t] - M[:, t])
    V_ub = pathwise.mean()
    se = pathwise.std(ddof=1) / sqrt(n)
    rho_res = t2_abs_sum / (t1_abs_sum + t2_abs_sum) if (t1_abs_sum + t2_abs_sum) > 0 else 0.0
    rho_denominator = t1_abs_by_path + t2_abs_by_path
    rho_denominator_mean = rho_denominator.mean()
    if n > 1 and rho_denominator_mean > 0:
        rho_influence = (t2_abs_by_path - rho_res * rho_denominator) / rho_denominator_mean
        rho_se = rho_influence.std(ddof=1) / sqrt(n)
        rho_ci_low = max(0.0, rho_res - 1.959963984540054 * rho_se)
        rho_ci_high = min(1.0, rho_res + 1.959963984540054 * rho_se)
    else:
        rho_se = np.nan
        rho_ci_low = np.nan
        rho_ci_high = np.nan
    if return_rho_stats:
        return V_ub, se, rho_res, rho_se, rho_ci_low, rho_ci_high
    return V_ub, se, rho_res


# ============================================================
#  Nested AB
# ============================================================
def nested_ab(S_out, G_out, betas, stats, sigma, n_inner, rng):
    n = S_out.shape[0]
    T = N_STEPS
    disc_f = np.exp(-R * np.arange(T + 1) * DT)
    Z_all = np.zeros((n, T + 1))
    Q = np.zeros((n, T + 1))
    for t in range(T + 1):
        Z_all[:, t] = payoff(G_out[:, t])
    for t in range(1, T + 1):
        if t < T and betas[t] is not None:
            C_hat = np.einsum(
                "ij,j->i", basis(S_out[:, t], G_out[:, t], stats), betas[t])
        else:
            C_hat = np.zeros(n)
        Q[:, t] = np.maximum(Z_all[:, t], C_hat)
    Q[:, 0] = Z_all[:, 0]

    M = np.zeros((n, T + 1))
    for t in range(1, T + 1):
        St, Gt = inner_step(S_out[:, t - 1], G_out[:, t - 1], t - 1, n_inner, sigma, rng)
        Z_in = payoff(Gt)
        if t < T and betas[t] is not None:
            C_in = np.einsum(
                "ij,j->i", basis(St.reshape(-1), Gt.reshape(-1), stats), betas[t]
            ).reshape(St.shape)
            E_Q = np.maximum(Z_in, C_in).mean(axis=1)
        else:
            E_Q = Z_in.mean(axis=1)
        dM = disc_f[t] * Q[:, t] - disc_f[t] * E_Q
        M[:, t] = M[:, t - 1] + dM
    pathwise = np.full(n, -np.inf)
    for t in range(T + 1):
        pathwise = np.maximum(pathwise, disc_f[t] * Z_all[:, t] - M[:, t])
    return pathwise.mean(), pathwise.std(ddof=1) / sqrt(n)


# ============================================================
#  Validation gate: European geo Asian MC vs closed form
# ============================================================
def validate_european(sigma, seed=0, n_mc=400000):
    rng = np.random.default_rng(seed)
    S, G = simulate_paths(n_mc, sigma, rng)
    eur_mc = exp(-R * T_YEARS) * payoff(G[:, N_STEPS]).mean()
    eur_se = exp(-R * T_YEARS) * payoff(G[:, N_STEPS]).std(ddof=1) / sqrt(n_mc)
    eur_cf = european_geo_asian_closed_form(sigma)
    return eur_cf, eur_mc, eur_se


# ============================================================
#  Train once, then run paired pilot/production stream bundles
# ============================================================
def train_policy(sigma, training_seed, strike=None):
    """Train and pre-select one policy for a problem setting."""
    global K
    if strike is not None:
        K = float(strike)
    clock = __import__("time").perf_counter
    started = clock()
    root = np.random.SeedSequence(int(training_seed))
    train_ss, select_ss = root.spawn(2)
    S_tr, G_tr = simulate_paths(
        N_TRAIN, sigma, np.random.default_rng(train_ss))
    stats = compute_std_stats(S_tr, G_tr)
    betas = lsm_train(S_tr, G_tr, stats)

    # This guard is fixed before every certification stream.
    V_european = european_geo_asian_closed_form(sigma)
    S_select, G_select = simulate_paths(
        N_SELECT, sigma, np.random.default_rng(select_ss))
    V_select, se_select, _ = lsm_lower_bound(
        S_select, G_select, betas, stats, return_stats=True)
    use_european = bool(V_select - 1.645 * se_select <= V_european)
    return {
        "sigma": float(sigma),
        "strike": float(K),
        "stats": stats,
        "betas": betas,
        "V_european": float(V_european),
        "selection_value": float(V_select),
        "selection_se": float(se_select),
        "european_fallback": use_european,
        "training_seed": int(training_seed),
        "training_seconds": clock() - started,
    }


def evaluate_trained_policy(
    trained,
    evaluation_seed,
    *,
    n_out=None,
    n_res=None,
    n_in=None,
    residual_mode="stratified",
):
    """Evaluate paired cheap and nested routes without retraining."""
    global K
    K = float(trained["strike"])
    sigma = float(trained["sigma"])
    n_out = N_OUT if n_out is None else int(n_out)
    n_res = N_RES if n_res is None else int(n_res)
    n_in = N_IN if n_in is None else int(n_in)
    root = np.random.SeedSequence(int(evaluation_seed))
    outer_ss, cheap_ss, nested_ss = root.spawn(3)
    clock = __import__("time").perf_counter

    started = clock()
    S_out, G_out = simulate_paths(
        n_out, sigma, np.random.default_rng(outer_ss))
    V_LB_policy = lsm_lower_bound(
        S_out, G_out, trained["betas"], trained["stats"])
    V_LB = (
        trained["V_european"]
        if trained["european_fallback"]
        else V_LB_policy
    )
    outer_lb_seconds = clock() - started

    started = clock()
    V_cons, se_cons, rho_res = hybrid_and_rho(
        S_out, G_out, trained["betas"], trained["stats"], sigma,
        n_res, np.random.default_rng(cheap_ss),
        residual_mode=residual_mode)
    cheap_seconds = clock() - started

    started = clock()
    V_nested, se_nested = nested_ab(
        S_out, G_out, trained["betas"], trained["stats"], sigma,
        n_in, np.random.default_rng(nested_ss))
    nested_seconds = clock() - started

    gap_h = (V_cons - V_LB) / V_LB * 100
    gap_n = (V_nested - V_LB) / V_LB * 100
    values = np.asarray([
        V_LB, V_cons, se_cons, V_nested, se_nested,
        gap_h, gap_n, rho_res,
    ])
    if not np.isfinite(values).all() or V_LB <= 0.0:
        raise FloatingPointError(
            f"invalid geometric-Asian evaluation {evaluation_seed}: {values}")
    return dict(
        seed=int(evaluation_seed), sigma=sigma, strike=K, V_LB=V_LB,
        V_LB_policy=V_LB_policy, V_european=trained["V_european"],
        selection_value=trained["selection_value"],
        selection_se=trained["selection_se"],
        european_fallback=trained["european_fallback"],
        V_cons=V_cons, se_cons=se_cons,
        V_nested=V_nested, se_nested=se_nested, rho_res=rho_res,
        gap_h=gap_h, gap_n=gap_n, residual_mode=residual_mode,
        n_out=n_out, n_res=n_res, n_in=n_in,
        outer_lb_seconds=outer_lb_seconds,
        cheap_seconds=cheap_seconds, nested_seconds=nested_seconds,
    )


def run_one_seed(seed, sigma, strike=None, residual_mode="stratified"):
    """Backward-compatible entry point using one independent setting bundle."""
    root = np.random.SeedSequence(int(seed))
    training_ss, evaluation_ss = root.spawn(2)
    trained = train_policy(
        sigma,
        int(training_ss.generate_state(1, dtype=np.uint64)[0]),
        strike=strike,
    )
    result = evaluate_trained_policy(
        trained,
        int(evaluation_ss.generate_state(1, dtype=np.uint64)[0]),
        residual_mode=residual_mode,
    )
    result["training_seconds"] = trained["training_seconds"]
    result["seed"] = int(seed)
    return result


if __name__ == "__main__":
    print("=" * 74)
    print("FAMILY A: Bermudan geometric-average Asian call (path-dependent, 2-D state)")
    print(f"S0={S0} K={K} r={R} T={T_YEARS} steps={N_STEPS} deg={DEGREE} "
          f"N_train={N_TRAIN} N_out={N_OUT} n_res={N_RES}")
    print("=" * 74)

    SIGMA_GRID = [float(x) for x in os.environ.get(
        "GEO_GRID", "0.10,0.20,0.30,0.40,0.60,0.80,1.00,1.20,1.50"
    ).split(",") if x.strip()]

    print("\n--- VALIDATION GATE: European geo-Asian  closed-form vs MC ---")
    validation_rows = []
    for sg in SIGMA_GRID:
        cf, mc, se = validate_european(sg, seed=12345)
        rel = abs(mc - cf) / cf * 100 if cf > 0 else float('nan')
        passed = bool(np.isfinite(rel) and rel < 0.5)
        flag = "OK" if passed else "FAIL"
        validation_rows.append(dict(
            sigma=sg, closed_form=cf, monte_carlo=mc, monte_carlo_se=se,
            relative_error_pct=rel, passed=passed, seed=12345, n_mc=400000))
        print(f"  sigma={sg:.2f}:  closed-form={cf:.4f}   MC={mc:.4f}±{se:.4f}   "
              f"rel_err={rel:.2f}%  [{flag}]")
    validation_output = os.environ.get(
        "GEO_VALIDATION_OUTPUT",
        os.path.join(os.path.dirname(__file__), "results",
                     "geoasian_european_validation.csv"))
    os.makedirs(os.path.dirname(validation_output), exist_ok=True)
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(validation_output, index=False)
    print(f"validation CSV: {validation_output}")
    if not bool(validation_frame["passed"].all()):
        raise RuntimeError(
            "geometric-Asian European validation exceeded 0.5% relative error")
    if os.environ.get("GEO_VALIDATE_ONLY", "0") == "1":
        raise SystemExit(0)

    print("\n--- MAIN: sigma sweep (5 seeds each) ---")
    all_rows = []
    for sg in SIGMA_GRID:
        seeds = list(range(1, N_SEEDS + 1))
        if WORKERS == 1:
            rows = [run_one_seed(s, sg) for s in seeds]
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                rows = list(pool.map(lambda s: run_one_seed(s, sg), seeds))
        all_rows.extend(rows)
        V_LB = np.mean([r['V_LB'] for r in rows])
        eur_cf = european_geo_asian_closed_form(sg)
        rho = np.mean([r['rho_res'] for r in rows])
        gh = np.array([r['gap_h'] for r in rows]); gn = np.array([r['gap_n'] for r in rows])
        Vc = np.mean([r['V_cons'] for r in rows]); Vn = np.mean([r['V_nested'] for r in rows])
        fallback = any(r["european_fallback"] for r in rows)
        mono = "European fallback active" if fallback else "V_LB>=Eur OK"
        feas = "E[UB]>=LB OK" if Vc >= V_LB - 1e-6 else "**UB<LB**"
        print(f"\n sigma={sg:.2f}:  Eur_geo(CF)={eur_cf:.3f}  V_LB={V_LB:.3f} [{mono}]")
        print(f"   rho_res={rho:.3f}  V_cons={Vc:.3f}  V_nested={Vn:.3f} [{feas}]")
        denom = sqrt(len(rows))
        print(f"   gap_hybrid={gh.mean():.2f}±{gh.std(ddof=1)/denom:.2f}%   "
              f"gap_nested={gn.mean():.2f}±{gn.std(ddof=1)/denom:.2f}%")

    print("\n" + "=" * 74)
    print("WITHIN-FAMILY rho_res -> gap relationship")
    print("=" * 74)
    import collections
    bysig = collections.defaultdict(list)
    for r in all_rows:
        bysig[r['sigma']].append(r)
    print(f"{'sigma':>6} {'rho_res':>8} {'gap_hyb%':>9} {'gap_nest%':>10}")
    for sg in SIGMA_GRID:
        rows = bysig[sg]
        print(f"{sg:>6.2f} {np.mean([r['rho_res'] for r in rows]):>8.3f} "
              f"{np.mean([r['gap_h'] for r in rows]):>9.2f} "
              f"{np.mean([r['gap_n'] for r in rows]):>10.2f}")

    output = os.environ.get(
        "GEO_OUTPUT", os.path.join(os.path.dirname(__file__), "results", "geoasian_9point_canonical.csv"))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output, index=False)
    print(f"CSV: {output}")
