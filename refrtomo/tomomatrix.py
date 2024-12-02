import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from scipy.sparse import csc_matrix


def _raytrace_vertical(s, r, dx, dz, ox, oz, nx, nz, x, z):
    """Vertical ray

    Compute vertical ray and associated tomographic matrix

    Parameters
    ----------
    s : :obj:`list`
        Source location
    r : :obj:`list`
        Receiver location
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    z : :obj:`np.ndarray`
        Vertical axis

    Returns
    -------
    R : :obj:`np.ndarray`
        Tomographic dense matrix
    c : :obj:`np.ndarray`
        Column indices (index of grid cells)
    v : :obj:`np.ndarray`
        Value (length of rays in grid cells)

    """
    # Define x-index and extreme z-indexes of ray
    xray, zray = s[0], s[1]
    ix = int((xray + dx // 2 - x[0]) // dx)
    izin = int((zray + dz // 2 - z[0]) // dz)
    izend = int((r[1] + dz // 2 - z[0]) // dz)

    # Create tomographic matrix
    R = np.zeros(nx * nz)
    if izend - izin > 0:
        R[ix * nz + izin] = z[izin] + dz / 2 - s[1] # source side cell
        R[ix * nz + izin + 1: ix * nz + izend] = dz # middle cells
        R[ix * nz + izend] = r[1] - z[izend] + dz / 2 # receiver side cell
    else:
        R[ix * nz + izin] = r[1] - s[1]
    c = np.arange(ix * nz + izin, ix * nz + izend + 1)
    v = np.array((r[1] - s[1],))
    if izend - izin > 0:
        v = np.hstack((z[izin] + dz / 2 - s[1],
                       dz * np.ones(izend - izin - 1),
                       r[1] - z[izend] + dz / 2))
    return R, c, v


def _raytrace_generic(s, r, dx, dz, ox, oz, nx, nz, x, z):
    """Generic ray

    Compute generic ray and associated tomographic matrix

    Parameters
    ----------
    s : :obj:`list`
        Source location
    r : :obj:`list`
        Receiver location
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    z : :obj:`np.ndarray`
        Vertical axis

    Returns
    -------
    R : :obj:`np.ndarray`
        Tomographic dense matrix
    c : :obj:`np.ndarray`
        Column indices (index of grid cells)
    v : :obj:`np.ndarray`
        Value (length of rays in grid cells)

    """
    # Define indices of source and parametric straight ray
    xray, zray = s[0], s[1]
    m = (r[1] - s[1]) / (r[0] - s[0])
    q = s[1] - m * s[0]

    # Create tomographic matrix
    R = np.zeros(nx * nz)
    c, v = np.zeros(3 * (nx + nz)), np.zeros(3 * (nx + nz))
    ic = 0

    while xray < r[0]:
        # find grid points of source
        ix, iz = int((xray + dx / 2 - x[0]) / dx), int(
            (zray + dz / 2 - z[0]) / dz)
        # computing z intersecting x-edge of the grid point
        xedge = x[ix] + dx / 2
        zedge = xedge * m + q
        # in case the ray finished its trajectory, stop iterations
        if xedge == xray:
            break
        if xedge > r[0]:
            xedge = r[0]
            zedge = r[1]
        izedge = int((zedge + dz / 2 - z[0]) / dz)
        if izedge == iz:
            # find length of ray from x to x-edge
            rray = sqrt((xedge - xray) ** 2 + (zedge - zray) ** 2)
            R[ix * nz + iz] = rray
            c[ic], v[ic] = ix * nz + iz, rray
            ic += 1
        elif izedge > iz:
            # next zedge above current z cell: split ray from x to x-edge into
            # multiple rays passing through different z-cells
            nzcells = izedge - iz + 1
            zend = (x[ix + 1] - dx / 2) * m + q
            if zend > r[1]:
                zend = r[1]
            zs = z[iz] + np.arange(nzcells - 1) * dz + dz // 2
            zs = np.insert(zs, 0, zray)
            zs = np.append(zs, zend)
            xs = (zs - q) / m
            rrays = np.sqrt(np.diff(xs) ** 2 + np.diff(zs) ** 2)
            R[ix * nz + iz: ix * nz + iz + nzcells] = rrays
            c[ic:ic + nzcells], v[ic:ic + nzcells] = np.arange(ix * nz + iz,
                                                               ix * nz + iz + nzcells), rrays
            ic += nzcells
        else:
            # next zedge below current z cell: split ray from x to x-edge into
            # multiple rays passing through different z-cells
            nzcells = iz - izedge + 1
            zend = (x[ix + 1] - dx / 2) * m + q
            if zend < r[1]:
                zend = r[1]
            zs = z[iz] - np.arange(nzcells - 1) * dz - dz / 2
            zs = np.insert(zs, 0, zray)
            zs = np.append(zs, zend)
            xs = (zs - q) / m
            rrays = np.sqrt(np.diff(xs) ** 2 + np.diff(zs) ** 2)
            R[ix * nz + iz - nzcells + 1: ix * nz + iz + 1] = rrays[::-1]
            c[ic:ic + nzcells], v[ic:ic + nzcells] = np.arange(
                ix * nz + iz - nzcells + 1, ix * nz + iz + 1), rrays[::-1]
            ic += nzcells
        xray, zray = xedge, zedge
    return R, c[:ic].astype(np.int64), v[:ic]


def raytrace_straight(s, r, dx, dz, ox, oz, nx, nz, x, z):
    """Raytrace

    Compute straight ray between two points and associated tomographic matrix.

    Parameters
    ----------
    s : :obj:`list`
        Source location
    r : :obj:`list`
        Receiver location
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    oz : :obj:`float`
        Vertical axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    x : :obj:`np.ndarray`
        Horizontal axis
    z : :obj:`np.ndarray`
        Vertical axis

    Returns
    -------
    R : :obj:`np.ndarray`
        Tomographic dense matrix
    c : :obj:`np.ndarray`
        Column indices (index of grid cells)
    v : :obj:`np.ndarray`
        Value (lenght of rays in grid cells)

    """
    if dx % 2. != 0 or dz % 2. != 0:
        raise ValueError('dx and dz must multiples of 2')
   
    # vertical ray
    if s[0] == r[0]:
        if r[1] > s[1]:
            # source below receiver
            return _raytrace_vertical(s, r, dx, dz, ox, oz, nx, nz, x, z)
        else:
            # source above receiver - swap them
            return _raytrace_vertical(r, s, dx, dz, ox, oz, nx, nz, x, z)

    # generic ray
    if r[0] > s[0]:
        # source on the left or receiver
        return _raytrace_generic(s, r, dx, dz, ox, oz, nx, nz, x, z)
    else:
        # source on the right or receiver - swap them
        return _raytrace_generic(r, s, dx, dz, ox, oz, nx, nz, x, z)

    
def tomographic_matrix(survey, dx, dz, ox, oz, nx, nz, x, z,
                       debug=False, plotflag=False, vel=None, figsize=(15, 3)):
    """Tomographich matrix

    Compute set of rays and associated tomographic matrix

    Parameters
    ----------
    survey : :obj:`list`
        Survey object
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    oz : :obj:`float`
        Vertical axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    x : :obj:`np.ndarray`
        Horizontal axis
    z : :obj:`np.ndarray`
        Vertical axis
    plotflag : :obj:`bool`, optional
        Plot rays and matrix coverage
    vel : :obj:`np.ndarray`, optional
        Velocity model

    Returns
    -------
    R : :obj:`scipy.sparse.csc_matrix`
        Tomographic sparse matrix

    """
    #R = []
    rows = []
    cols = []
    v = []
    for iray, ray in enumerate(survey):
        R_ = [raytrace_straight(ray.ray[i], ray.ray[i+1], dx, dz, ox, oz, nx, nz, x, z)[1:]
              for i in range(ray.ray.shape[0]-1)]
        ctmp, vtmp = np.hstack([r[0] for r in R_]), np.hstack([r[1] for r in R_])
        #R.append(np.sum([r[0] for r in R_], axis=0))
        rows.append(iray * np.ones_like(ctmp))
        cols.append(ctmp)
        v.append(vtmp)
    #R = np.vstack(R)
    R = csc_matrix((np.hstack(v), (np.hstack(rows), np.hstack(cols))), shape=(len(v), nx * nz))  
    if debug: print(f'tomographic_matrix: {R.shape[0]} rows, {R.shape[1]} columns')
  
    if plotflag:
        plt.figure(figsize=figsize)
        plt.imshow(vel.T, cmap='jet', extent =(x[0], x[-1], z[-1], z[0]))
        plt.colorbar()

        for ray in survey:
            plt.plot(ray.ray[:,0], ray.ray[:,1], 'w')
            plt.scatter(ray.src[0], ray.src[1], marker='*', s=150, c='r', edgecolors='k')
            plt.scatter(ray.rec[0], ray.rec[1], marker='v', s=200, c='w', edgecolors='k')
            plt.axis('tight');
        
        Rcove = np.sum(R.toarray(), axis=0).reshape(nx, nz).T
        plt.figure(figsize=figsize)
        plt.imshow(Rcove, vmin=0, vmax=0.1*Rcove.max(), extent=(x[0], x[-1], z[-1], z[0]))
        # Show the colorbar
        plt.colorbar()
        plt.axis('tight')
        plt.title('Ray coverage')
    
    return R