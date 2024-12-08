import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.signal import filtfilt
from pylops.optimization.leastsquares import *
from refrtomo.invert import *
from refrtomo.survey import *

#%% Velocity Model

nx, nz = 1360, 280
dx, dz = 2., 2.

vel_true = np.fromfile("../data/marmousi.bin", dtype='float32', sep="")
vel_true = vel_true.reshape(nx, nz)[100:501, 100:181]#[500:901, 100:181]
nx, nz = vel_true.shape
x, z = np.arange(nx) * dx, np.arange(nz) * dz

# Smooth velocity
nsmooth = 15
vel_sm = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_true, axis=0)
vel_sm = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_sm, axis=1)

# Initial velocity
nsmooth = 20
vel_init = vel_sm.copy()
for _ in range(50):
    vel_init = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_init, axis=1)

# vel_init = np.tile(1700 + 10 * z, (nx, 1))
vel_init = np.tile(1700 + 5 * z, (nx, 1))

## plt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create a 1x3 grid of subplots
# True velocity
im = axes[0].imshow(vel_true.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
axes[0].set_title('True Velocity')
axes[0].set_xlabel('x [m]')
axes[0].set_ylabel('y [m]')
axes[0].set_ylim(z[-1], z[0])
axes[0].axis('tight')
# Smooth velocity
im = axes[1].imshow(vel_sm.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
axes[1].set_title('Smooth Velocity')
axes[1].set_xlabel('x [m]')
axes[1].set_ylabel('y [m]')
axes[1].set_ylim(z[-1], z[0])
axes[1].axis('tight')
# Init velocity
im = axes[2].imshow(vel_init.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
axes[2].set_title('Init Velocity')
axes[2].set_xlabel('x [m]')
axes[2].set_ylabel('y [m]')
axes[2].set_ylim(z[-1], z[0])
axes[2].axis('tight')
# Add a colorbar for all subplots
# cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
for ax, ix in zip(axs, [nx//4, nx//2, 3*nx//4]):
    ax.plot(vel_sm[ix], z, 'k')
    ax.plot(vel_init[ix], z, 'r')
axs[-1].invert_yaxis()
plt.show()

#%% Geometry

geometry = 'Surface'
ns = 20
nr = 10
s = np.empty((2, ns))
r = np.empty((2, nr))
max_x = x[-1]
max_z = z[-1]
if geometry == 'Crosshole':
    # Crosshole
    s[0, :ns] = 5 * dx
    s[1, :ns] = np.linspace(5 * dz, max_z - 5 * dz, ns)
    r[0, :nr] = max_x - 5 * dx
    r[1, :nr] = np.linspace(5 * dz, max_z - 5 * dz, nr)

elif geometry == 'VSP':
    # VSP
    s[0, :ns] = np.linspace(7.5 * dx, max_x - 7.5 * dx, ns)
    s[1, :ns] = 2.5 * dz
    r[0, :nr] = max_x / 2
    r[1, :nr] = np.linspace(2.5 * dz, max_z - 2.5 * dz, nr)

elif geometry == 'Surrounded':
    # Surrounded
    s[0, :ns//2] = 5 * dx
    s[1, :ns//2] = np.linspace(5 * dz, max_z - 5 * dz, ns//2)
    r[0, :nr//2] = max_x - 5 * dx
    r[1, :nr//2] = np.linspace(5 * dz, max_z - 5 * dz, nr//2)

    s[1, ns//2:] = 5 * dz
    s[0, ns//2:] = np.linspace(5 * dx, max_x - 5 * dx, ns//2)
    r[1, nr//2:] = max_z - 5 * dz
    r[0, nr//2:] = np.linspace(5 * dx, max_x - 5 * dx, nr//2)

elif geometry == 'Surface':
    # Surface
    r[0, :] = np.linspace(4 * dx, max_x - 4 * dx, nr)
    r[1, :] = 0
    s[0, :] = np.linspace(4 * dx, max_x - 4 * dx, ns)
    s[1, :] = 0


plt.figure()
im = plt.imshow(vel_init.T, cmap='jet', origin='lower', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.colorbar(im)
plt.scatter(r[0], r[1], c='black', s=100, marker='s')
plt.scatter(s[0], s[1], c='r', s=100, marker='*')
# plt.xticks(x-dx//2)
# plt.yticks(z-dx//2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Geometry', fontsize=15, fontweight='bold')
plt.grid('on', which='both')
plt.axis('tight')
# plt.axis('scaled')
plt.show()



survey = survey_geom(s, r, minoffset=50)


fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
isrc_values = [0, 1, 2, 3]  # Adjust these indices to fit your survey data

for idx, ax in enumerate(axes.flat):
    isrc = isrc_values[idx]  # Get the current isrc value
    ax.scatter(survey[isrc].src[0], survey[isrc].src[1], c='r', label='Source')
    ax.scatter(survey[isrc].rec[0], survey[isrc].rec[1], c='b', label='Receiver')
    ax.set_title(f"Source Index: {isrc}")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    # ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

#%% Inverse crime
lmax = 1e3
nl = 2000
thetas = np.hstack([np.arange(-85, -40, 0.1), np.arange(40, 85, 0.1)])
avasurvey = survey_raytrace(survey, vel_sm.T, x, z, lmax, nl, thetas, dzout=5., ray_rec_mindistance=5., debug=True)

# Display observed traveltimes
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
for isrc in range(ns):
    display_survey_tobs(avasurvey, s[0][isrc], ax=ax)
plt.show()

Robs = tomographic_matrix(avasurvey, dx, dz, 0, 0, nx, nz, x, z, plotflag=True, vel=vel_sm)

tobs = extract_tobs(avasurvey)
tobs_R = Robs @ (1/vel_sm.ravel())


plt.figure()
im = plt.imshow( vel_sm.T, cmap='jet', extent = (x[0], x[-1], z[-1], z[0]))
plt.colorbar(im)
for ss in avasurvey:
    src = ss.src
    plt.scatter(src[0], src[1], marker='*', s=150, c='r', edgecolors='k')
    plt.scatter(ss.rec[0], ss.rec[1], marker='v', s=200, c='w', edgecolors='k')

    for irec, rec in enumerate(ss.rec):
        plt.plot(ss.ray[:,0], ss.ray[:,1], 'w')
    plt.axis('tight')
plt.show()

plt.figure(figsize=(15, 3))
plt.plot(tobs, 'r')
plt.plot(tobs_R, 'k')
plt.show()


#%% Initial rays and traveltimes

initsurvey = survey_raytrace(survey, vel_init.T, x, z, lmax, nl, thetas, dzout=5., ray_rec_mindistance=5., debug=True)

# Match surveys
avasurvey_matched, initsurvey_matched = match_surveys(avasurvey, initsurvey, debug=True)

# Tomographic matrix and traveltimes
Rinit = tomographic_matrix(initsurvey_matched, dx, dz, 0, 0, nx, nz, x, z,
                           plotflag=True, vel=vel_init, figsize=(15, 3))

tobs = extract_tobs(avasurvey_matched)
tobs_init = extract_tobs(initsurvey_matched)
tinit = Rinit @ (1/vel_init.ravel())

plt.figure(figsize=(15, 3))
plt.plot(tobs, 'r')
plt.plot(tobs_init, 'k')
plt.plot(tinit, '--g');


#%% regularized_inversion

Dop = Laplacian((nx, nz), weights=(10, 1))
slowninv = regularized_inversion(MatrixMult(Rinit),
                                 tobs, [Dop, ], epsRs=[2e2, ],
                                 x0=1. / vel_init.ravel(),
                                 **dict(iter_lim=100, damp=1e-1))[0]
vel_inv = 1. / (slowninv.reshape(nx, nz) + 1e-10)
tinv = Rinit @ (1 / vel_inv.ravel())

plt.figure()
im = plt.imshow(vel_inv.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'), plt.ylabel('y [m]')
plt.title('Inverted velocity')
plt.ylim(z[-1], z[0])
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_sm[ix], z, 'k', label = 'Smoothed')
    ax.plot(vel_init[ix], z, 'r', label = 'Initial')
    ax.plot(vel_inv[ix], z, 'g', label = 'Inverted')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label = 'Smoothed')
plt.plot(tobs_init, 'k', label = 'Initial')
plt.plot(tinv, '--g', label = 'Inverted')
plt.legend()
plt.show()

#%% Naive Gauss-Newton method

ngniter = 5
# first version, updating slow with full tobs at each step
Dop = Laplacian((nx, nz), weights=(10, 1))
vel_inv = vel_init.copy()
misfit = []
tin = time.time()
for iiter in range(ngniter):
    print(f'Iteration {iiter + 1}/{ngniter}')
    # Raytrace in initial model
    invsurvey = survey_raytrace(survey, vel_inv.T, x, z, lmax, nl, thetas,
                                dzout=5., ray_rec_mindistance=5., debug=True)

    # Match surveys
    avasurvey_matched, invsurvey_matched = match_surveys(avasurvey, invsurvey, debug=True)

    # Tomographic matrix and traveltimes
    Rinv = tomographic_matrix(invsurvey_matched, dx, dz, 0, 0, nx, nz, x, z,
                              plotflag=True, vel=vel_inv, figsize=(15, 3))
    tobs = extract_tobs(avasurvey_matched)
    tinv = Rinv @ (1. / vel_inv.ravel())
    plt.figure(figsize=(15, 3))
    plt.plot(tobs, 'k')
    plt.plot(tinv, 'r')
    misfit.append(np.linalg.norm(tobs - tinv) / len(tobs))

    # Invert slowness model (just few iterations to avoid overfitting at each step)
    slowninv = regularized_inversion(MatrixMult(Rinv),
                                     tobs, [Dop, ], epsRs=[8e1, ],
                                     x0=1. / vel_inv.ravel(),
                                     **dict(iter_lim=40, damp=1e-1))[0]
    vel_inv = 1. / (slowninv.reshape(nx, nz) + 1e-10)
tend = time.time()
print(f'Elapsed time: {tend - tin} sec')


plt.figure()
im = plt.imshow(vel_inv.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'), plt.ylabel('y [m]')
plt.title('Inverted velocity')
plt.ylim(z[-1], z[0])
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_sm[ix], z, 'k', label = 'Smoothed')
    ax.plot(vel_init[ix], z, 'r', label = 'Initial')
    ax.plot(vel_inv[ix], z, 'g', label = 'Inverted')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label = 'Smoothed')
plt.plot(tobs_init, 'k', label = 'Initial')
plt.plot(tinv, '--g', label = 'Inverted')
plt.legend()
plt.show()

plt.figure()
plt.plot(misfit, 'k')
plt.title('Misfit');
plt.show()

#%% Naive Gauss-Newton method 2nd


Dop = Laplacian((nx, nz), weights=(10, 1))
# second version, updating dslow with full dtobs at each step
vel_inv = vel_init.copy()
slown_inv = 1. / vel_inv.ravel()
misfit = []
tin = time.time()
for iiter in range(ngniter):
    print(f'Iteration {iiter + 1}/{ngniter}')
    # Raytrace in initial model
    invsurvey = survey_raytrace(survey, vel_inv.T, x, z, lmax, nl, thetas, dzout=5.,
                                ray_rec_mindistance=5., debug=True)

    # Match surveys
    avasurvey_matched, invsurvey_matched = match_surveys(avasurvey, invsurvey, debug=True)

    # Tomographic matrix and traveltimes
    Rinv = tomographic_matrix(invsurvey_matched, dx, dz, 0, 0, nx, nz, x, z,
                              plotflag=False, vel=vel_inv, debug=True)
    tobs = extract_tobs(avasurvey_matched)
    tinv = Rinv @ slown_inv

    # Residual data
    dtobs = tobs - tinv
    misfit.append(np.linalg.norm(dtobs) / len(tobs))
    # plt.figure(figsize=(15, 3))
    # plt.plot(tobs, 'k')
    # plt.plot(tinv, 'r')

    # Invert slowness update (just few iterations to avoid overfitting at each step)
    dslown_inv = regularized_inversion(MatrixMult(Rinv),
                                       dtobs, [Dop, ],
                                       epsRs=[8e1, ], dataregs=[-Dop * slown_inv.ravel(), ],
                                       **dict(iter_lim=40, damp=1e-1))[0]
    slown_inv += dslown_inv
    vel_inv = 1. / (slown_inv.reshape(nx, nz) + 1e-10)
tend = time.time()
print(f'Elapsed time: {tend - tin} sec')


plt.figure()
im = plt.imshow(vel_inv.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'), plt.ylabel('y [m]')
plt.title('Inverted velocity')
plt.ylim(z[-1], z[0])
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_sm[ix], z, 'k', label = 'Smoothed')
    ax.plot(vel_init[ix], z, 'r', label = 'Initial')
    ax.plot(vel_inv[ix], z, 'g', label = 'Inverted')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label = 'Smoothed')
plt.plot(tobs_init, 'k', label = 'Initial')
plt.plot(tinv, '--g', label = 'Inverted')
plt.legend()
plt.show()

plt.figure()
plt.plot(misfit, 'k')
plt.title('Misfit');
plt.show()

#%% Proper Gauss-Newton method
RTomo = RefrTomo(survey, avasurvey, x, z, lmax, nl, thetas, dzout=5.,
                 ray_rec_mindistance=5., epsL=8e1, weightsL=(10, 1), returnJ=True,
                 debug=True)
vel_inv, misfit = RTomo.solve(vel_init, 5, damp=1e-1, lsqr_args=dict(iter_lim=40, show=True))


plt.figure()
im = plt.imshow(vel_inv.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'), plt.ylabel('y [m]')
plt.title('Inverted velocity')
plt.ylim(z[-1], z[0])
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_sm[ix], z, 'k', label = 'Smoothed')
    ax.plot(vel_init[ix], z, 'r', label = 'Initial')
    ax.plot(vel_inv[ix], z, 'g', label = 'Inverted')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label = 'Smoothed')
plt.plot(tobs_init, 'k', label = 'Initial')
plt.plot(tinv, '--g', label = 'Inverted')
plt.legend()
plt.show()

plt.figure()
plt.plot(misfit, 'k')
plt.title('Misfit');
plt.show()

#%% Diagonal Preconditioner


# Compute diagonal of the normal matrix (approximate preconditioner)
diag_P = (Rinit.T @ Rinit).diagonal()  # Extract the diagonal directly
diag_P = np.diag(diag_P)

# Create the preconditioner
P = 1 / (diag_P + 1e-10)  # Add a small value to avoid division by zero


slowninv = preconditioned_inversion(MatrixMult(Rinit),
                                 tobs, P,
                                 x0=1. / vel_init.ravel(),
                                 **dict(iter_lim=100, damp=1e-1))[0]
vel_inv = 1. / (slowninv.reshape(nx, nz) + 1e-10)
tinv = Rinit @ (1 / vel_inv.ravel())


plt.figure()
im = plt.imshow(vel_inv.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'), plt.ylabel('y [m]')
plt.title('Inverted velocity')
plt.ylim(z[-1], z[0])
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_sm[ix], z, 'k', label = 'Smoothed')
    ax.plot(vel_init[ix], z, 'r', label = 'Initial')
    ax.plot(vel_inv[ix], z, 'g', label = 'Inverted')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label = 'Smoothed')
plt.plot(tobs_init, 'k', label = 'Initial')
plt.plot(tinv, '--g', label = 'Inverted')
plt.legend()
plt.show()