import numpy as np

from scipy.sparse.linalg import LinearOperator as spLinearOperator
from pylops.basicoperators import *
from refrtomo.raytrace import raytrace
from refrtomo.survey import *
from refrtomo.tomomatrix import *


class RefrTomo:
    def __init__(self, survey, surveydata, x, z, lmax, nl, thetas, 
                 dzout=1., ray_rec_mindistance=1., epsL=1., weightsL=(1, 1),
                 returnJ=False, debug=False):
        self.survey, self.surveydata = survey, surveydata
        self.x, self.z = x, z
        self.lmax = lmax
        self.nl = nl
        self.thetas = thetas
        self.dzout = dzout
        self.ray_rec_mindistance = ray_rec_mindistance
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
        if self.debug: print('fun: Computing...')
            
        # Save most recent x
        self.xold = x.copy()
        
        # Reshape x into a velocity model
        vel = 1. / x.reshape(self.dims)
        
        plt.figure(figsize=(10,5))
        im = plt.imshow(vel.T, cmap='jet', vmin=1800, vmax=3000)
        plt.axis('tight')
        plt.xlabel('x [m]'),plt.ylabel('y [m]')
        
        # Raytrace in velocity model
        invsurvey = survey_raytrace(self.survey, vel.T, self.x, self.z, self.lmax, self.nl, 
                                    self.thetas, dzout=self.dzout, 
                                    ray_rec_mindistance=self.ray_rec_mindistance)

        # Match surveys
        surveydata_matched, invsurvey_matched = match_surveys(self.surveydata, invsurvey)
        
        # Tomographic matrix and traveltimes
        Rinv = tomographic_matrix(invsurvey_matched, self.dx, self.dz, 0, 0, self.nx, self.nz, 
                                  self.x, self.z, plotflag=True, vel=vel)
        tobs = extract_tobs(surveydata_matched)
        
        # Add smoothing regularization term
        Rinv = VStack([MatrixMult(Rinv), self.epsL * self.Lop]).todense()
        self.Rinvold = Rinv
        
        # Add smoothing regularization term
        tobs = np.pad(tobs, (0, self.nx*self.nz))
        tinv = Rinv @ x
        
        if not self.returnJ:
            return tobs - tinv
        else:
            return tobs - tinv, Rinv
    
    def jacobian(self, x):
        if self.xold is not None and np.allclose(x, self.xold):
            # use Jacobian already computed in fun
            if self.debug: print('jacobian: Use stored Jacobian')
            return self.Rinvold
        else:
            if self.debug: print('jacobian: Recompute Jacobian')
            
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
                                      self.x, self.z, plotflag=debug, vel=vel)

            # Add smoothing regularization term
            Rinv = VStack([MatrixMult(Rinv), self.epsL * self.Lop]).todense()
            
            return Rinv