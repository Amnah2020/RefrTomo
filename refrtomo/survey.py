import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from refrtomo.raytrace import *
import cupy as cp
from joblib import Parallel, delayed

Ray = namedtuple('Ray', ['src', 'rec', 'tobs', 'ray'])


def survey_geom(srcs, recs, minoffset=0):
    survey = []
    for s in srcs.T:
        ioff = np.where(np.abs(s[0]-recs[0]) > minoffset)[0]
        survey.append(Ray(s, recs[:, ioff], tobs=None, ray=None)) 
    return survey


def survey_geom_observed(srcs, recs, tobs, minoffset=0):
    survey = []
    for isrc, s in enumerate(srcs.T):
        for irec, r in enumerate(recs.T):
            if np.abs(s[0]-r[0]) > minoffset and not np.isnan(tobs[isrc, irec]):
                survey.append(Ray(s, r, tobs=tobs[isrc, irec], ray=None)) 
    return survey


def survey_raytrace_optimized(
        survey, vel, x, z, lmax, nl, thetas, dzout=0.1, ray_rec_mindistance=1.0, debug=False, tolerance_z=1.0, match_direct=False
):
    dx, dz = x[1] - x[0], z[1] - z[0]

    def process_source(s):
        src = s.src
        # Raytrace
        rays, rays_turning, thetas_turning = raytrace(
            vel, x, z, dx, dz, lmax, nl, src, thetas, dzout=.1, debug=False
        )
        # In case we want to do direct rays as in crosshole
        matched_rays = []
        if match_direct:
            for rec in s.rec.T:
                for ray in rays:
                    if np.allclose(ray[-1, :2], rec, atol=ray_rec_mindistance):
                        matched_rays.append(
                            Ray(src, rec, tobs=ray[-1, -1], ray=ray[:, :2])
                        )
            return matched_rays

        for rec in s.rec.T:
            if rec[1] == 0:
                # Receiver at z=0
                rays_endx = np.array([ray[-1, 0] for ray in rays_turning])
                if rays_endx.size == 0:
                    if debug:
                        print(f"No valid rays found for receiver at {rec}.")
                    continue
                iray = np.argmin(np.abs(rays_endx - rec[0]))
                ray_rec_distance = rays_endx[iray] - rec[0]
                if np.abs(ray_rec_distance) < ray_rec_mindistance:
                    matched_rays.append(
                        Ray(src, rec, tobs=rays_turning[iray][-1, -1], ray=rays_turning[iray][:, :2])
                    )
            else:
                # Receiver not at z=0
                ray_trunc = []
                for ray in rays_turning:
                    z_rec_ray = np.abs(ray[:, 1] - rec[1]) <= tolerance_z
                    if len(np.where(z_rec_ray)[0]) > 0:
                        ray_trunc_idx = np.where(z_rec_ray)[0][-1]
                        ray_trunc.append(ray[:ray_trunc_idx + 1, :])

                rays_endx = np.array([ray[-1, 0] for ray in ray_trunc])
                if rays_endx.size == 0:
                    if debug:
                        print(f"No valid rays found for receiver at {rec}.")
                    continue
                iray = np.argmin(np.abs(rays_endx - rec[0]))
                ray_rec_distance = rays_endx[iray] - rec[0]
                if np.abs(ray_rec_distance) < ray_rec_mindistance:
                    matched_rays.append(
                        Ray(src, rec, tobs=ray_trunc[iray][-1, -1], ray=ray_trunc[iray][:, :2])
                    )
        return matched_rays

    # Parallel processing for each source
    avasurvey_results = Parallel(n_jobs=-1)(
        delayed(process_source)(s) for s in survey
    )

    # Flatten the results
    avasurvey = [ray for sublist in avasurvey_results for ray in sublist]

    if debug:
        nsr = np.sum([s.rec.shape[1] for s in survey])
        print(
            f"survey_raytrace_optimized: {nsr} Source-receiver pairs in survey, {len(avasurvey)} Source-receiver paired with ray..."
        )
    return avasurvey


def survey_raytrace(survey, vel, x, z, lmax, nl, thetas, dzout=.1, ray_rec_mindistance=1., debug=False, tolerance_z=1., match_direct=False):
    dx, dz = x[1] - x[0], z[1] - z[0]
    
    avasurvey = []
    for s in survey:
        src = s.src
        # Raytrace
        rays, rays_turning, thetas_turning = raytrace(vel, x, z, dx, dz, lmax, nl, src, thetas, dzout=.1, debug=False)

        # In case we want to do direct rays as in crosshole
        if match_direct:
            for rec in s.rec.T:
                rays_endx = np.array([ray[-1, 0] for ray in rays])
                iray = np.argmin(np.abs(rays_endx - rec[0]))
                ray_rec_distance = rays_endx[iray] - rec[0]
                if np.abs(ray_rec_distance) < ray_rec_mindistance:
                    avasurvey.append(Ray(src, rec, tobs=rays[iray][-1, -1], ray=rays[iray][:, :2]))
            return avasurvey

        for rec in s.rec.T:
            if rec[1] == 0:
                rays_endx = np.array([ray[-1, 0] for ray in rays_turning])
                iray = np.argmin(np.abs(rays_endx - rec[0]))
                ray_rec_distance = rays_endx[iray] - rec[0]
                if np.abs(ray_rec_distance) < ray_rec_mindistance:
                    avasurvey.append(Ray(src, rec, tobs=rays_turning[iray][-1, -1], ray=rays_turning[iray][:, :2]))
            else:
                ray_trunc = []
                for ray in rays_turning:
                    z_rec_ray = np.abs(ray[:, 1] - rec[1]) <= tolerance_z
                    if len(np.where(z_rec_ray)[0]) > 0:
                        ray_trunc_idx = np.where(z_rec_ray)[0][-1]
                        ray_trunc.append(ray[:ray_trunc_idx + 1, :])

                rays_endx = np.array([ray[-1, 0] for ray in ray_trunc])
                if len(rays_endx) == 0:
                    print("No valid rays found for this source-receiver pair.")
                    continue
                iray = np.argmin(np.abs(rays_endx - rec[0]))
                ray_rec_distance = rays_endx[iray] - rec[0]
                if np.abs(ray_rec_distance) < ray_rec_mindistance:
                    avasurvey.append(Ray(src, rec, tobs=ray_trunc[iray][-1, -1], ray=ray_trunc[iray][:, :2]))

    if debug: 
        nsr = np.sum([s.rec.shape[1] for s in survey])
        print(f'survey_raytrace: {nsr} Source-receiver pairs in survey, {len(avasurvey)} Source-receiver paired with ray...') 
    return avasurvey


def extract_tobs(survey):
    tobs = [ray.tobs for ray in survey]
    return tobs


def match_surveys(survey1, survey2, debug=False):
    survey1matched = []
    survey2matched = []
    
    # extract src and rec from first survey
    survey1_srcrec = []
    for iray in range(len(survey1)):
        survey1_srcrec.append(list(np.hstack((survey1[iray].src, survey1[iray].rec))))
    for iray in range(len(survey2)):
        srcrec = list(np.hstack((survey2[iray].src, survey2[iray].rec)))
        if srcrec in survey1_srcrec:
            survey1matched.append(survey1[survey1_srcrec.index(srcrec)])
            survey2matched.append(survey2[iray])
    if debug: 
        print(f'match_surveys: {len(survey1)} Rays in survey1, {len(survey2)} Rays in survey2, {len(survey1matched)} Matched rays...')
    
    return survey1matched, survey2matched


def display_survey(survey):
    for isrc, surv in enumerate(survey):
        plt.scatter(survey[isrc].src[0], survey[isrc].src[1] + isrc * 5, c='r')
        plt.scatter(survey[isrc].rec[0], survey[isrc].rec[1] + isrc * 5 , c='b')
        
    
def display_survey_rays(survey, vel, x, z, sx=None, figsize=(15, 12)):
    plt.figure(figsize=figsize)
    for s in survey:
        src = s.src
        if sx is None or sx == src[0]:
            plt.imshow(vel, cmap='jet', extent = (x[0], x[-1], z[-1], z[0]))
            plt.scatter(src[0], src[1], marker='*', s=150, c='r', edgecolors='k')
            plt.scatter(s.rec[0], s.rec[1], marker='v', s=200, c='w', edgecolors='k')

            for irec, rec in enumerate(s.rec):
                plt.plot(s.ray[:,0], s.ray[:,1], 'w')
            plt.axis('tight')
            

def display_survey_tobs(survey, sx, c='k', ax=None, figsize=(15, 12)):
    recs_isrc = []
    tobs_isrc = []
    for s in survey:
        if s.src[0] == sx:
            recs_isrc.append(s.rec[0])
            tobs_isrc.append(s.tobs)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(recs_isrc, tobs_isrc, f'.-{c}')
    ax.invert_yaxis()