import numpy as np
from scipy.sparse.linalg import lsqr


def gauss_newton(f, x0, niter, damp, lsqr_args={}):
    
    x = x0.copy()
    misfit = []
    for iiter in range(niter):
        print(f'Iteration {iiter+1}/{niter}')
        d, J = f(x)
        misfit.append(np.linalg.norm(d) / len(d))

        # Invert
        dx = lsqr(J, d, damp=damp, **lsqr_args)[0]
        x += dx
    
    return x, misfit