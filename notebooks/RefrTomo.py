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

vel_actual = np.fromfile("../data/marmousi.bin", dtype='float32', sep="")
vel_actual = vel_actual.reshape(nx, nz)[100:501, 100:181]
nx, nz = vel_actual.shape
x, z = np.arange(nx) * dx, np.arange(nz) * dz

# Smoothed velocity
nsmooth = 15
vel_smooth = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_actual, axis=0)
vel_smooth = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_smooth, axis=1)

# Initial velocity
nsmooth = 20
vel_initial = vel_smooth.copy()
for _ in range(50):
    vel_initial = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, vel_initial, axis=1)

vel_initial = np.tile(1700 + 5 * z, (nx, 1))

# Plot Velocity Models
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Actual velocity
im = axes[0].imshow(vel_actual.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
axes[0].set_title('Actual Velocity')
axes[0].set_xlabel('x [m]')
axes[0].set_ylabel('y [m]')
axes[0].set_ylim(z[-1], z[0])
axes[0].axis('tight')

# Smoothed velocity
im = axes[1].imshow(vel_smooth.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
axes[1].set_title('Smoothed Velocity')
axes[1].set_xlabel('x [m]')
axes[1].set_ylabel('y [m]')
axes[1].set_ylim(z[-1], z[0])
axes[1].axis('tight')

# Initial velocity
im = axes[2].imshow(vel_initial.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
axes[2].set_title('Initial Velocity')
axes[2].set_xlabel('x [m]')
axes[2].set_ylabel('y [m]')
axes[2].set_ylim(z[-1], z[0])
axes[2].axis('tight')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
for ax, ix in zip(axs, [nx//4, nx//2, 3*nx//4]):
    ax.plot(vel_smooth[ix], z, 'k', label = 'Smoothed Velocity')
    ax.plot(vel_initial[ix], z, 'r', label = 'Initial Velocity')
axs[-1].invert_yaxis()
plt.legend()
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
    s[0, :ns] = 5 * dx
    s[1, :ns] = np.linspace(5 * dz, max_z - 5 * dz, ns)
    r[0, :nr] = max_x - 5 * dx
    r[1, :nr] = np.linspace(5 * dz, max_z - 5 * dz, nr)

elif geometry == 'VSP':
    s[0, :ns] = np.linspace(7.5 * dx, max_x - 7.5 * dx, ns)
    s[1, :ns] = 2.5 * dz
    r[0, :nr] = max_x / 2
    r[1, :nr] = np.linspace(2.5 * dz, max_z - 2.5 * dz, nr)

elif geometry == 'Surrounded':
    s[0, :ns//2] = 5 * dx
    s[1, :ns//2] = np.linspace(5 * dz, max_z - 5 * dz, ns//2)
    r[0, :nr//2] = max_x - 5 * dx
    r[1, :nr//2] = np.linspace(5 * dz, max_z - 5 * dz, nr//2)

    s[1, ns//2:] = 5 * dz
    s[0, ns//2:] = np.linspace(5 * dx, max_x - 5 * dx, ns//2)
    r[1, nr//2:] = max_z - 5 * dz
    r[0, nr//2:] = np.linspace(5 * dx, max_x - 5 * dx, nr//2)

elif geometry == 'Surface':
    r[0, :] = np.linspace(4 * dx, max_x - 4 * dx, nr)
    r[1, :] = 0
    s[0, :] = np.linspace(4 * dx, max_x - 4 * dx, ns)
    s[1, :] = 0

plt.figure()
im = plt.imshow(vel_initial.T, cmap='jet', origin='lower', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
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

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
isrc_values = [0, 1, 2, 3]
for idx, ax in enumerate(axes.flat):
    isrc = isrc_values[idx]
    ax.scatter(survey[isrc].src[0], survey[isrc].src[1], c='r', label='Source')
    ax.scatter(survey[isrc].rec[0], survey[isrc].rec[1], c='b', label='Receiver')
    ax.set_title(f"Source Index: {isrc}")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True)
plt.tight_layout()
plt.show()

#%% Inverse crime
lmax = 1e3
nl = 2000
thetas = np.hstack([np.arange(-85, -40, 0.1), np.arange(40, 85, 0.1)])
avasurvey = survey_raytrace(survey, vel_smooth.T, x, z, lmax, nl, thetas, dzout=5., ray_rec_mindistance=5., debug=True)

# Display observed traveltimes
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
for isrc in range(ns):
    display_survey_tobs(avasurvey, s[0][isrc], ax=ax)
plt.title('Observed Travel Times')
plt.show()

Robs = tomographic_matrix(avasurvey, dx, dz, 0, 0, nx, nz, x, z, plotflag=False, vel=vel_smooth)

tobs = extract_tobs(avasurvey)
tobs_R = Robs @ (1/vel_smooth.ravel())


plt.figure()
im = plt.imshow( vel_smooth.T, cmap='jet', extent = (x[0], x[-1], z[-1], z[0]))
plt.colorbar(im)
for ss in avasurvey:
    src = ss.src
    plt.scatter(src[0], src[1], marker='*', s=150, c='r', edgecolors='k')
    plt.scatter(ss.rec[0], ss.rec[1], marker='v', s=200, c='w', edgecolors='k')

    for irec, rec in enumerate(ss.rec):
        plt.plot(ss.ray[:,0], ss.ray[:,1], 'w')
plt.axis('auto')
plt.title('Smoothed velocity and Ray coverage')
plt.show()

plt.figure(figsize=(15, 3))
plt.plot(tobs, 'r', label= "Acutal picks")
plt.plot(tobs_R, 'k', label = "R picks")
plt.legend()
plt.show()


#%% Initial rays and traveltimes

initsurvey = survey_raytrace(survey, vel_initial.T, x, z, lmax, nl, thetas, dzout=5., ray_rec_mindistance=5., debug=True)

# Match surveys
avasurvey_matched, initsurvey_matched = match_surveys(avasurvey, initsurvey, debug=True)

# Tomographic matrix and traveltimes
Rinit = tomographic_matrix(initsurvey_matched, dx, dz, 0, 0, nx, nz, x, z,
                           plotflag=False, vel=vel_initial, figsize=(15, 3))

tobs = extract_tobs(avasurvey_matched)
tobs_init = extract_tobs(initsurvey_matched)
tinit = Rinit @ (1/vel_initial.ravel())

plt.figure(figsize=(15, 3))
plt.plot(tobs, 'r', label = 'Smoothed')
plt.plot(tobs_init, 'k', label = 'Initial')
plt.plot(tinit, '--g', label = 'R')
plt.legend()
plt.show()


#%% Regularized Inversion
Dop = Laplacian((nx, nz), weights=(10, 1))
slowninv_regularized = regularized_inversion(MatrixMult(Rinit),
                                             tobs, [Dop, ], epsRs=[2e2, ],
                                             x0=1. / vel_initial.ravel(),
                                             **dict(iter_lim=100, damp=1e-1))[0]
vel_regularized = 1. / (slowninv_regularized.reshape(nx, nz) + 1e-10)
tinv_regularized = Rinit @ (1 / vel_regularized.ravel())

plt.figure()
im = plt.imshow(vel_regularized.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.xlabel('x [m]'), plt.ylabel('y [m]')
plt.title('Regularized Velocity')
plt.ylim(z[-1], z[0])
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_smooth[ix], z, 'k', label='Smoothed')
    ax.plot(vel_initial[ix], z, 'r', label='Initial')
    ax.plot(vel_regularized[ix], z, 'g', label='Regularized')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure(figsize=(15, 3))
plt.plot(tobs, 'r', label='Smoothed')
plt.plot(tobs_init, 'k', label='Initial')
plt.plot(tinv_regularized, '--g', label='Regularized')
plt.legend()
plt.show()

#%% Naive Gauss-Newton Method

ngniter = 5
# first version, updating slow with full tobs at each step
Dop = Laplacian((nx, nz), weights=(10, 1))
vel_gauss_newton_1 = vel_initial.copy()
misfit_gauss_newton_1 = []
tin = time.time()
for iiter in range(ngniter):
    print(f'Iteration {iiter + 1}/{ngniter}')
    invsurvey = survey_raytrace(survey, vel_gauss_newton_1.T, x, z, lmax, nl, thetas,
                                dzout=5., ray_rec_mindistance=5., debug=True)

    avasurvey_matched, invsurvey_matched = match_surveys(avasurvey, invsurvey, debug=True)

    Rinv = tomographic_matrix(invsurvey_matched, dx, dz, 0, 0, nx, nz, x, z,
                              plotflag=False, vel=vel_gauss_newton_1, figsize=(15, 3))
    tobs = extract_tobs(avasurvey_matched)
    tinv_gauss_newton_1 = Rinv @ (1. / vel_gauss_newton_1.ravel())
    plt.figure(figsize=(15, 3))
    plt.plot(tobs, 'k')
    plt.plot(tinv_gauss_newton_1, 'r')
    misfit_gauss_newton_1.append(np.linalg.norm(tobs - tinv_gauss_newton_1) / len(tobs))

    slowninv = regularized_inversion(MatrixMult(Rinv),
                                     tobs, [Dop, ], epsRs=[8e1, ],
                                     x0=1. / vel_gauss_newton_1.ravel(),
                                     **dict(iter_lim=40, damp=1e-1))[0]
    vel_gauss_newton_1 = 1. / (slowninv.reshape(nx, nz) + 1e-10)
tend = time.time()
print(f'Elapsed time: {tend - tin} sec')

plt.figure()
im = plt.imshow(vel_gauss_newton_1.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.title('Naive Gauss-Newton Velocity (1st)')
plt.axis('tight')
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_smooth[ix], z, 'k', label='Smoothed')
    ax.plot(vel_initial[ix], z, 'r', label='Initial')
    ax.plot(vel_gauss_newton_1[ix], z, 'g', label='Naive Gauss-Newton (1st)')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
# plt.plot(tobs, 'r', label='Smoothed')
plt.plot(tobs_init, 'k', label='Initial')
plt.plot(tinv_gauss_newton_1, '--g', label='Naive Gauss-Newton (1st)')
plt.legend()
plt.show()

plt.figure()
plt.plot(misfit_gauss_newton_1, 'k')
plt.title('Misfit: Naive Gauss-Newton (1st)')
plt.show()

#%% Naive Gauss-Newton Method 2nd

Dop = Laplacian((nx, nz), weights=(10, 1))
vel_gauss_newton_2 = vel_initial.copy()
slown_inv = 1. / vel_gauss_newton_2.ravel()
misfit_gauss_newton_2 = []
tin = time.time()
for iiter in range(ngniter):
    print(f'Iteration {iiter + 1}/{ngniter}')
    invsurvey = survey_raytrace(survey, vel_gauss_newton_2.T, x, z, lmax, nl, thetas, dzout=5.,
                                ray_rec_mindistance=5., debug=True)

    avasurvey_matched, invsurvey_matched = match_surveys(avasurvey, invsurvey, debug=True)

    Rinv = tomographic_matrix(invsurvey_matched, dx, dz, 0, 0, nx, nz, x, z,
                              plotflag=False, vel=vel_gauss_newton_2, debug=True)
    tobs = extract_tobs(avasurvey_matched)
    tinv_gauss_newton_2 = Rinv @ slown_inv

    dtobs = tobs - tinv_gauss_newton_2
    misfit_gauss_newton_2.append(np.linalg.norm(dtobs) / len(tobs))

    dslown_inv = regularized_inversion(MatrixMult(Rinv),
                                       dtobs, [Dop, ],
                                       epsRs=[8e1, ], dataregs=[-Dop * slown_inv.ravel(), ],
                                       **dict(iter_lim=40, damp=1e-1))[0]
    slown_inv += dslown_inv
    vel_gauss_newton_2 = 1. / (slown_inv.reshape(nx, nz) + 1e-10)
tend = time.time()
print(f'Elapsed time: {tend - tin} sec')

plt.figure()
im = plt.imshow(vel_gauss_newton_2.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.title('Naive Gauss-Newton Velocity (2nd)')
plt.axis('tight')
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_smooth[ix], z, 'k', label='Smoothed')
    ax.plot(vel_initial[ix], z, 'r', label='Initial')
    ax.plot(vel_gauss_newton_2[ix], z, 'g', label='Naive Gauss-Newton (2nd)')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label='Smoothed')
plt.plot(tobs_init, 'k', label='Initial')
plt.plot(tinv_gauss_newton_2, '--g', label='Naive Gauss-Newton (2nd)')
plt.legend()
plt.show()

plt.figure()
plt.plot(misfit_gauss_newton_2, 'k')
plt.title('Misfit: Naive Gauss-Newton (2nd)')
plt.show()

#%% Proper Gauss-Newton Method

RTomo = RefrTomo(survey, avasurvey, x, z, lmax, nl, thetas, dzout=5.,
                 ray_rec_mindistance=5., epsL=8e1, weightsL=(10, 1), returnJ=True,
                 debug=True)
vel_proper_gauss, misfit_proper_gauss = RTomo.solve(vel_initial, 5, damp=1e-1, lsqr_args=dict(iter_lim=40, show=True))


tinv_gauss_newton_pr = Rinv @ (1/vel_proper_gauss.ravel())

plt.figure()
im = plt.imshow(vel_proper_gauss.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.axis('tight')
plt.title('Proper Gauss-Newton Velocity')
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_smooth[ix], z, 'k', label='Smoothed')
    ax.plot(vel_initial[ix], z, 'r', label='Initial')
    ax.plot(vel_proper_gauss[ix], z, 'g', label='Proper Gauss-Newton')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label='Smoothed')
plt.plot(tobs_init, 'k', label='Initial')
plt.plot(tinv_gauss_newton_pr, '--g', label='Proper Gauss-Newton')
plt.legend()
plt.show()

plt.figure()
plt.plot(misfit_proper_gauss, 'k')
plt.title('Misfit: Proper Gauss-Newton')
plt.show()

#%% comparison

# Comparison of Travel Times (tinv) from Each Method

plt.figure(figsize=(15, 5))
plt.plot(tobs, 'k', label='Observed Travel Times')
plt.plot(tinv_regularized, '--r', label='Regularized')
plt.plot(tinv_gauss_newton_1, '--g', label='Gauss-Newton (1st)')
plt.plot(tinv_gauss_newton_2, '--p', label='Gauss-Newton (2nd)')
plt.plot(tinv_gauss_newton_pr, '--y', label='Gauss-Newton proper')
# plt.plot(tinv_preconditioned, '--m', label='Preconditioned')
plt.xlabel('Receiver Index')
plt.ylabel('Travel Time')
plt.title('Comparison of Travel Times (tinv) from Each Method')
plt.legend()
plt.grid(True)
plt.show()

# Comparison of Misfit from Each Method

plt.figure(figsize=(15, 5))
plt.plot(misfit_gauss_newton_1, '-g', label='Gauss-Newton (1st)')
plt.plot(misfit_gauss_newton_2, '-b', label='Gauss-Newton (2nd)')
plt.plot(misfit_proper_gauss, '-r', label='Proper Gauss-Newton')
plt.title('Comparison of Misfit from Each Method')
plt.xlabel('Iteration')
plt.ylabel('Misfit')
plt.legend()
plt.grid(True)
plt.show()

#%% Local save

# import pickle
#
# np.save("RefrTomo_temp/vel_regularized.npy", vel_regularized)
# np.save("RefrTomo_temp/vel_gauss_newton_1.npy", vel_gauss_newton_1)
# np.save("RefrTomo_temp/vel_gauss_newton_2.npy", vel_gauss_newton_2)
# np.save("RefrTomo_temp/vel_proper_gauss.npy", vel_proper_gauss)
#
# # Save heterogeneous objects using pickle
# with open("RefrTomo_temp/avasurvey.pkl", "wb") as f:
#     pickle.dump(avasurvey, f)
#
# with open("RefrTomo_temp/initsurvey.pkl", "wb") as f:
#     pickle.dump(initsurvey, f)



# # To load them back
# with open("RefrTomo_temp/avasurvey.pkl", "rb") as f:
#     avasurvey_loaded = pickle.load(f)
#
# with open("RefrTomo_temp/initsurvey.pkl", "rb") as f:
#     initsurvey_loaded = pickle.load(f)


#%% Diagonal Preconditioner

diag_P = (Rinit.T @ Rinit).diagonal()
diag_P = np.diag(diag_P)

P = 1 / (diag_P + 1e-10)

slowninv_preconditioned = preconditioned_inversion(MatrixMult(Rinit),
                                                   tobs, P, show=True,
                                                   x0=1. / vel_initial.ravel(),
                                                   **dict(iter_lim=100, damp=1e-1))[0]
vel_preconditioned = 1. / (slowninv_preconditioned.reshape(nx, nz) + 1e-10)
tinv_preconditioned = Rinit @ (1 / vel_preconditioned.ravel())

plt.figure()
im = plt.imshow(vel_preconditioned.T, cmap='jet', vmin=1800, vmax=3000, extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(r[0], r[1], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(s[0], s[1], marker='*', s=150, c='r', edgecolors='k')
plt.colorbar(im)
plt.title('Preconditioned Velocity')
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
for ax, ix in zip(axs, [nx // 4, nx // 2, 3 * nx // 4]):
    ax.plot(vel_smooth[ix], z, 'k', label='Smoothed')
    ax.plot(vel_initial[ix], z, 'r', label='Initial')
    ax.plot(vel_preconditioned[ix], z, 'g', label='Preconditioned')
axs[-1].invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.plot(tobs, 'r', label='Smoothed')
plt.plot(tobs_init, 'k', label='Initial')
plt.plot(tinv_preconditioned, '--g', label='Preconditioned')
plt.legend()
plt.show()