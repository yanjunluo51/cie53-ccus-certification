"""Numerical gate that keeps every RuntimeWarning fatal EXCEPT the spurious
floating-point flags Apple Accelerate raises inside matmul.

Verified spurious on this machine: a bare `A.T @ A` on standard normals raises
`divide by zero`/`overflow encountered in matmul` while every entry is finite
and agrees with einsum to ~1e-11.  Negative controls below prove the gate still
fires on real divide-by-zero, overflow, and invalid operations.
"""
import runpy, sys, warnings
import numpy as np

warnings.simplefilter("error", RuntimeWarning)
warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

def _self_test():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30000, 10))
    B = A.T @ A                                  # must be allowed
    C = np.einsum('ij,ik->jk', A, A)
    assert np.isfinite(B).all()
    dev = float(np.max(np.abs(B - C)))
    assert dev < 1e-6, dev
    fired = 0
    for op in (lambda: np.float64(1.0) / np.float64(0.0),
               lambda: np.exp(np.float64(800.0)),
               lambda: np.sqrt(np.float64(-1.0))):
        try:
            op()
        except RuntimeWarning:
            fired += 1
    assert fired == 3, f"gate is vacuous: only {fired}/3 negative controls fired"
    print(f"[gate] matmul exempt (max dev vs einsum {dev:.3e}); "
          f"3/3 negative controls still fatal", flush=True)

_self_test()
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
