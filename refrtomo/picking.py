import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, medfilt
from matplotlib.patches import Rectangle


def firstarrival_picking(x, t, d, ireczeroff, nrecnearoff, 
                         threshnear=1e-3, threshfar=1e-1, 
                         nmed=None, nsmooth=None, 
                         forcezerooff=False, maxdt=None, clip=1):
    
    nrec = d.shape[1]
    tmax = np.zeros(nrec)
    tfirst = np.zeros(nrec)
    
    for irec in range(nrec):
        dmax = np.abs(d[:, irec]).max()
        itmax = np.argmin(np.abs(np.abs(d[:, irec]) - dmax))
        if np.abs(irec - ireczeroff) < nrecnearoff:
            itfirst = np.where(np.abs(d[:, irec])>= threshnear*dmax)[0][0]
        else:
            itfirst = np.where(np.abs(d[:, irec])>= threshfar*dmax)[0][0]
        tmax[irec] = t[itmax]
        tfirst[irec] = t[itfirst]
    
    tfirst_unproc = tfirst.copy()
    tmask = np.full(nrec, False)
    if forcezerooff:
        tfirst[ireczeroff] = 0.
    if nmed is not None:    
        tfirst = medfilt(tfirst, nsmooth)
    if maxdt is not None:
        dtfist = np.pad(np.diff(tfirst), (0, 1), mode='edge') 
        tmask = np.abs(dtfist) > maxdt 
    if nsmooth is not None:
        tfirst = filtfilt(np.ones(nsmooth)/nsmooth, 1, tfirst)
    tfirst[tmask] = np.nan
    
    plt.figure(figsize=(15, 5))
    plt.imshow(d, cmap='gray', vmin=-clip*d.max(), vmax=clip*d.max(), extent=(x[0], x[-1], t[-1], t[0]))
    plt.axvline(x[ireczeroff], color='w')
    if ireczeroff-nrecnearoff > 0:
        plt.axvline(x[ireczeroff-nrecnearoff], color='w', linestyle='--')
    if ireczeroff+nrecnearoff < nrec:
        plt.axvline(x[ireczeroff+nrecnearoff], color='w', linestyle='--')
    plt.axis('tight')
    plt.xlabel('Offset [m]')
    plt.ylabel('T [s]')
    plt.title('Data')
    plt.ylim(t[-1], t[0])
    plt.grid(which='both')
    plt.colorbar()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(d, cmap='gray', vmin=-clip*d.max(), vmax=clip*d.max(), extent=(x[0], x[-1], t[-1], t[0]))
    plt.axvline(x[ireczeroff], color='w')
    if ireczeroff-nrecnearoff > 0:
        plt.axvline(x[ireczeroff-nrecnearoff], color='w', linestyle='--')
    if ireczeroff+nrecnearoff < nrec:
        plt.axvline(x[ireczeroff+nrecnearoff], color='w', linestyle='--')
    plt.plot(x, tmax, 'b', lw=1)
    plt.plot(x, tfirst_unproc, '--r', lw=1)
    plt.plot(x, tfirst, 'r', lw=4)
    plt.axis('tight')
    plt.xlabel('Offset [m]')
    plt.ylabel('T [s]')
    plt.title('Data')
    plt.ylim(t[-1], t[0])
    plt.grid(which='both')
    plt.colorbar()
    
    return tfirst


def firstarrival_picking1(x, t, d, ireczeroff, ntwin, nxmin, thresh=1.,
                         nmed=None, nsmooth=None, 
                         forcezerooff=False, maxdt=None, clip=1):
    
    nt, nrec = d.shape
    tmax = np.zeros(nrec)
    tfirst = np.zeros(nrec)
    
    # find window away from zero offset to compute average amplitude
    twin = (0, ntwin)
    if ireczeroff < nrec // 2:
        rwin = (nrec-nxmin, nrec)
    else:
        rwin = (0, nxmin)
    dwin = d[twin[0]:twin[1], rwin[0]:rwin[1]]
    dwinave = np.mean(np.abs(dwin))
    dmask = np.abs(d) > thresh * dwinave
    
    """
    plt.figure(figsize=(15, 5))
    plt.imshow(dwin, cmap='gray', vmin=-clip*d.max(), vmax=clip*d.max())
    plt.axis('tight')
    plt.grid(which='both')
    plt.colorbar()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(d, cmap='gray', vmin=-clip*d.max(), vmax=clip*d.max())
    plt.axvline(x[ireczeroff], color='w')
    plt.axis('tight')
    plt.xlabel('Offset [m]')
    plt.ylabel('T [s]')
    plt.title('Data')
    plt.grid(which='both')
    plt.colorbar()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(dmask, cmap='gray')
    plt.axvline(x[ireczeroff], color='w')
    plt.axis('tight')
    plt.xlabel('Offset [m]')
    plt.ylabel('T [s]')
    plt.title('Mask')
    plt.grid(which='both')
    plt.colorbar()
    """
    
    for irec in range(nrec):
        itfirst = np.where(np.abs(d[:, irec]) > thresh * dwinave )[0]
        tfirst[irec] = t[-1 if len(itfirst) == 0 else itfirst[0]]
    
    tfirst_unproc = tfirst.copy()
    tmask = np.full(nrec, False)
    if forcezerooff:
        tfirst[ireczeroff] = 0.
    if nmed is not None:    
        tfirst = medfilt(tfirst, nsmooth)
    if maxdt is not None:
        dtfist = np.pad(np.diff(tfirst), (0, 1), mode='edge') 
        tmask = np.abs(dtfist) > maxdt 
    if nsmooth is not None:
        tfirst = filtfilt(np.ones(nsmooth)/nsmooth, 1, tfirst)
    tfirst[tmask] = np.nan
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    im = ax.imshow(d, cmap='gray', vmin=-clip*d.max(), vmax=clip*d.max(), extent=(x[0], x[-1], t[-1], t[0]))
    ax.add_patch(Rectangle((x[rwin[0]], t[twin[0]]), x[rwin[1]-1]-x[rwin[0]], t[twin[1]-1]-t[twin[0]], 
                 edgecolor='red', facecolor='none', lw=4))
    ax.axvline(x[ireczeroff], color='w')
    ax.axis('tight')
    ax.set_xlabel('Offset [m]')
    ax.set_ylabel('T [s]')
    ax.set_title('Data')
    ax.set_ylim(t[-1], t[0])
    ax.grid(which='both')
    plt.colorbar(im, ax=ax)
    

    plt.figure(figsize=(15, 5))
    plt.imshow(dmask, cmap='gray', extent=(x[0], x[-1], t[-1], t[0]))
    plt.axvline(x[ireczeroff], color='w')
    plt.axis('tight')
    plt.xlabel('Offset [m]')
    plt.ylabel('T [s]')
    plt.title('Mask')
    plt.ylim(t[-1], t[0])
    plt.grid(which='both')
    plt.colorbar()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(d, cmap='gray', vmin=-clip*d.max(), vmax=clip*d.max(), extent=(x[0], x[-1], t[-1], t[0]))
    plt.axvline(x[ireczeroff], color='w')
    plt.plot(x, tmax, 'b', lw=1)
    plt.plot(x, tfirst_unproc, '--r', lw=1)
    plt.plot(x, tfirst, 'r', lw=4)
    plt.axis('tight')
    plt.xlabel('Offset [m]')
    plt.ylabel('T [s]')
    plt.title('Data')
    plt.ylim(t[-1], t[0])
    plt.grid(which='both')
    plt.colorbar()
    
    return tfirst