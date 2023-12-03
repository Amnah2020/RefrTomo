import numpy as np
from math import sin, cos
from scipy.ndimage import uniform_filter
from scipy.io import loadmat
from scipy.integrate import solve_ivp


def rhsf(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
    """RHS of raytracing ODE 
    
    Parameters
    ----------
    l : indipendent variable l
    r : dependent variable containing (x, z, px, pz, t)
    slowness : slowness 2d model
    dsdx : horizontal derivative of slowness 2d model
    dsdz : vertical derivative of slowness 2d model
    xaxis : horizontal axis
    zaxis : vertical axis
    dx : horizontal spacing
    dz : vertical spacing

    Returns
    -------
    drdt : RHS evaluation
    
    """
    m, n = slowness.shape
    
    # extract the different terms of the solution
    x = r[0]
    z = r[1]
    px = r[2]
    pz = r[3]
    drdt = np.zeros(len(r))

    # identify current position of the ray in the model
    xx = int((x - xaxis[0]) / dx)
    zz = int((z - zaxis[0]) / dz)
    xx = min([xx, n-1])
    xx = max([xx, 1])
    zz = min([zz, m-1])
    zz = max([zz, 1]) 

    # extract s, ds/dx, ds/dz at current position (nearest-neighbour interpolation)
    s = slowness[round(zz), round(xx)]
    dsdx = dsdx[round(zz), round(xx)]
    dsdz = dsdz[round(zz), round(xx)]
    
    # evaluate RHS
    drdt[0] = px / s
    drdt[1] = pz / s
    drdt[2] = dsdx
    drdt[3] = dsdz
    drdt[4] = s
    return drdt


def raytrace(vel, xaxis, zaxis, dx, dz, lmax, nl, source, thetas, dzout=1.):
    """Raytracing for multiple rays defined by the initial conditions (source, thetas)
    
    Parameters
    ----------
    vel : np.ndarray
        2D Velocity model (nz x nx)
    xaxis : np.ndarray
        Horizonal axis 
    zaxis : np.ndarray
        Vertical axis 
    dx : float
        Horizonal spacing 
    dz : float
        Vertical spacing 
    lmax : float
        Max lenght of ray
    nl : int
        Number of steps of ray
    source : tuple
        Source location
    thetas : tuple
        Take-off angles
    
    """
    # Events
    def event_left(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
        return r[0]-xaxis[0]
    def event_right(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
        return xaxis[-1]-r[0]
    def event_top(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
        return r[1]-zaxis[0] + dzout
    def event_bottom(l, r, slowness, dsdx, dsdz, xaxis, zaxis, dx, dz):
        return zaxis[-1]-r[1]

    event_left.terminal = True # set to True to trigger termination as soon as the condition is met
    event_left.direction = -1 # set to -1 if we want to stop when going from positive to negative outputs of event
    event_right.terminal = True # set to True to trigger termination as soon as the condition is met
    event_right.direction = -1 # set to -1 if we want to stop when going from positive to negative outputs of event
    event_top.terminal = True # set to True to trigger termination as soon as the condition is met
    event_top.direction = -1 # set to -1 if we want to stop when going from positive to negative outputs of event
    event_bottom.terminal = True # set to True to trigger termination as soon as the condition is met
    event_bottom.direction = -1 # set to -1 if we want to stop when going from positive to negative outputs of event
    
    # Trajectory array
    lstep = np.linspace(0, lmax, nl)
    dl = lstep[1]

    # Source indices
    dz, dx = zaxis[1] - zaxis[0], xaxis[1] - xaxis[0]
    ixs, izs = int((source[0]-xaxis[0])/dx), int((source[1]-zaxis[0])/dz)
    
    # Slowness and its spatial derivatives
    slowness = 1./vel
    [dsdz, dsdx] = np.gradient(slowness, dz, dx)
    
    # Trace rays
    rays = []
    for theta in thetas:
        # Initial condition
        r0=[source[0], source[1], 
            sin(theta * np.pi / 180) / vel[izs, ixs],
            cos(theta * np.pi / 180) / vel[izs, ixs], 0]

        # Solve ODE
        sol = solve_ivp(rhsf, [lstep[0], lstep[-1]], r0, t_eval=lstep, 
                        args=(slowness, dsdx, dsdz, xaxis, zaxis, dx, dz), 
                        events=[event_left, event_right,
                                event_top, event_bottom])
        ray = sol['y'].T
        
        # Add ray (but first check that is not going out - events may not always catch it)
        if np.max(ray[:, 1]) < zaxis[-1] and np.min(ray[:, 0]) > xaxis[0] and np.max(ray[:, 0]) < xaxis[-1]:
            # Force last point to be at z=0 if event_top triggered (also check
            irayneg = np.where(ray[:, 1] < 0)[0]
            #if len(sol['t_events'][2]) > 0:
            if len(irayneg) > 0:
                irayposlast = irayneg[0] - 1
                irayneg = irayneg[0]
                rayx_z0 = ray[irayposlast, 0] + ray[irayposlast, 1] * dl / (ray[irayposlast, 1] - ray[irayneg, 1])
                ray = ray[:irayneg]
                ray[-1, 0], ray[-1, 1]  = rayx_z0, 0
        rays.append(ray)
    
    # Prune rays (remove those not arriving back)
    thetas_turning = []
    rays_turning = []
    for iray, ray in enumerate(rays):
        if ray[-1, 1] == 0.:
            rays_turning.append(ray)
            thetas_turning.append(thetas[iray])
    
    return rays, rays_turning, thetas_turning