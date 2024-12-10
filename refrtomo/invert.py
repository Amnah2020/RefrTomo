import time
import numpy as np

from scipy.sparse.linalg import LinearOperator as spLinearOperator
from pylops.basicoperators import *
from refrtomo.gaussnewton import gauss_newton
from refrtomo.raytrace import raytrace
from refrtomo.survey import *
from refrtomo.tomomatrix import *

class RefrTomo:
    def __init__(self, survey, surveydata, x, z, lmax, nl, thetas, 
                 dzout=1., ray_rec_mindistance=1., tolerance_z=1., epsL=1., weightsL=(1, 1),
                 returnJ=False, debug=False):
        self.survey, self.surveydata = survey, surveydata
        self.x, self.z = x, z
        self.lmax = lmax
        self.nl = nl
        self.thetas = thetas
        self.dzout = dzout
        self.ray_rec_mindistance = ray_rec_mindistance
        self.tolerance_z = tolerance_z
        self.returnJ = returnJ
        self.debug = debug

        # dimensions of velocity model as (nx, nz)
        self.nx, self.nz = len(x), len(z)
        self.dims = self.nx, self.nz
        # sampling of x and z axes
        self.dx, self.dz = x[1] - x[0], z[1] - z[0]
        # save most recent x and Jacobian
        self.xold, self.Rinvold = None, None
        # laplacian
        self.epsL = epsL
        self.Lop = Laplacian(self.dims, weights=weightsL)
        
    def fun(self, x):
        if self.debug: 
            tin = time.time()
            print('RefrTomo-fun: Computing...')
            
        # Save most recent x
        self.xold = x.copy()
        
        # Reshape x into a velocity model
        vel = 1. / x.reshape(self.dims)
        
        if self.debug:
            plt.figure(figsize=(15, 3))
            plt.imshow(vel.T, cmap='jet')
            plt.axis('tight')
            plt.xlabel('x [m]'),plt.ylabel('y [m]')

        # Raytrace in velocity model
        invsurvey = survey_raytrace(self.survey, vel.T, self.x, self.z, self.lmax, self.nl, 
                                    self.thetas, dzout=self.dzout, 
                                    ray_rec_mindistance=self.ray_rec_mindistance, 
                                    tolerance_z=self.tolerance_z, debug=self.debug)

        # Match surveys
        surveydata_matched, invsurvey_matched = match_surveys(self.surveydata, invsurvey, debug=self.debug)
        
        self.surveydata_matched = surveydata_matched
        self.invsurvey_matched = invsurvey_matched
        
        # Tomographic matrix and traveltimes
        Rinv = tomographic_matrix(invsurvey_matched, self.dx, self.dz, 0, 0, self.nx, self.nz, 
                                  self.x, self.z, debug=self.debug, plotflag=self.debug, vel=vel)
        tobs = extract_tobs(surveydata_matched)

        self.tobs = tobs
        
        # Add smoothing regularization term
        Rinv = VStack([MatrixMult(Rinv), self.epsL * self.Lop])
        self.Rinvold = Rinv
        
        # Add smoothing regularization term
        ntobs = len(tobs)
        tobs = np.pad(tobs, (0, self.nx*self.nz))
        tinv = Rinv @ x

        self.tinv = tinv[:ntobs]

        if self.debug: 
            tend = time.time()
            print(f'RefrTomo-fun: Misfit {np.linalg.norm(tobs[:ntobs] - tinv[:ntobs]) / ntobs:.4f}')
            print(f'RefrTomo-fun: Elapsed time {tend-tin} s...')
        
        if not self.returnJ:
            return tobs - tinv
        else:
            return tobs - tinv, Rinv
    
    def jacobian(self, x):
        if self.xold is not None and np.allclose(x, self.xold):
            # use Jacobian already computed in fun
            if self.debug: print('RefrTomo-jacobian: Use stored Jacobian')
            return self.Rinvold
        else:
            if self.debug: print('RefrTomo-jacobian: Recompute Jacobian')
            
            # Reshape x into a velocity model
            vel = 1. / x.reshape(self.dims)

            # Raytrace in velocity model
            invsurvey = survey_raytrace(self.survey, vel.T, self.x, self.z, self.lmax, self.nl, 
                                        self.thetas, dzout=self.dzout, 
                                        ray_rec_mindistance=self.ray_rec_mindistance)

            # Match surveys
            surveydata_matched, invsurvey_matched = match_surveys(self.surveydata, invsurvey)

            # Tomographic matrix and traveltimes
            Rinv = tomographic_matrix(invsurvey_matched, self.dx, self.dz, 0, 0, self.nx, self.nz, 
                                      self.x, self.z, plotflag=self.debug, vel=vel)

            # Add smoothing regularization term
            Rinv = VStack([MatrixMult(Rinv), self.epsL * self.Lop])
            
            return Rinv
        
    def solve(self, v0, niter, damp, lsqr_args={}):
        s0 = 1./v0.ravel()
        tin = time.time()
        sinv, misfit = gauss_newton(self.fun, s0, niter, damp, lsqr_args=lsqr_args)
        tend = time.time()
        vinv= 1. / (sinv.reshape(self.dims) + 1e-10)
        print(f'Elapsed time: {tend-tin} sec')
        return vinv, misfit
        