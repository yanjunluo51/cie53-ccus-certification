"""
FINAL variance-rate verification on a single documented instance:
CCUS single-regime (sigma_P=0.278, degree 3, eps=0.05, n_res=50), fixed policy
(train once, seed 7777). For each N_out, K=60 independent sampling reps;
Var(V_cons | beta_hat) should scale ~ 1/N_out. Fit log(Var) ~ log(N_out).
Supersedes the legacy 5-point exp_mse_rate_v2.csv (non-reproducible, gap
inconsistent with the current canonical). REAL CODE; no fabricated numbers.
"""
import sys, functools, numpy as np, csv
sys.path.insert(0,"drivers"); sys.path.insert(0,"shared")
import single_regime_ccus_driver as srd
from sample_splitting import SampleSplittingRNGs, simulate_inner_one_step
from hybrid_ab_conservative import hybrid_nestless_AB
from ccus_core.core.ccus_model import simulate_paths
np.seterr(all="ignore")
base=srd.make_params(); p=srd.make_single_regime_params(base); payoff_fn=srd.payoff_fn_for(p)
mp={"Q_regime":np.asarray(p.Q_regime),"dt":p.dt,"r":p.r,"mu_P_by_regime":np.asarray(p.mu_P_by_regime),"mu_C":p.mu_C,"sigma_P":p.sigma_P,"sigma_C":p.sigma_C}
tr=SampleSplittingRNGs.from_seed(7777)
ret=srd.lsm_3d_stdlog(p,30000,srd.DEGREE,seed=tr.training.bit_generator.seed_seq,eps=srd.EPS)
V_LB,betas_pr,std_stats=ret[0],ret[1],ret[2]
bh=srd.adapt_betas(betas_pr); sd=dict(zip(["mu_log_P","sigma_log_P","mu_log_C","sigma_log_C"],std_stats))
isim=functools.partial(simulate_inner_one_step,dt=p.dt,sigma_P=p.sigma_P,mu_P_by_regime=np.asarray(p.mu_P_by_regime),sigma_C=p.sigma_C,mu_C=p.mu_C,rho=0.0,Q_regime=np.asarray(p.Q_regime))
def vcons(nout,s,nres=50):
    rngs=SampleSplittingRNGs.from_seed(s); P,C,R=simulate_paths(p,nout,rng=rngs.outer)
    rh=np.random.default_rng(rngs.inner.bit_generator.seed_seq.spawn(1)[0])
    return hybrid_nestless_AB(outer_paths={"P":P,"C":C,"R":R},inner_simulator=isim,inner_rng=rh,beta_hat=bh,basis_powers=srd.basis_powers,standardisation_stats=sd,model_params=mp,n_res=nres,payoff_fn=payoff_fn,conservative=True)["V_UB"]
K=60; GRID=[1000,2000,5000,10000,20000,30000,50000,70000,100000]
print(f"FINAL VARIANCE RATE (single-regime CCUS, fixed policy, K={K})  V_LB={V_LB:.6e}")
print(f"{'N_out':>8} {'Var(Vcons)':>14} {'mean_Vcons':>14} {'gap%':>7}")
logN=[];logV=[];rows=[]
for nout in GRID:
    vs=np.array([vcons(nout,s) for s in range(6000,6000+K)])
    var=vs.var(ddof=1);m=vs.mean();gap=(m-V_LB)/V_LB*100
    print(f"{nout:>8} {var:>14.4e} {m:>14.4e} {gap:>6.1f}%",flush=True)
    rows.append((nout,var,m));logN.append(np.log(nout));logV.append(np.log(var))
A=np.vstack([np.array(logN),np.ones(len(logN))]).T
sl,ic=np.linalg.lstsq(A,np.array(logV),rcond=None)[0]
pred=A@np.array([sl,ic]);r2=1-np.sum((np.array(logV)-pred)**2)/np.sum((np.array(logV)-np.mean(logV))**2)
print(f"\nlog-log slope = {sl:.3f}  (ideal -1.000)   R^2={r2:.3f}")
with open("results/varrate_single_regime_ccus.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow(["N_out","Var_Vcons","mean_Vcons"]);[w.writerow(r) for r in rows]
print("CSV written.")
