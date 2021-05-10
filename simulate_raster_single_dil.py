#!/usr/bin/python3
import sys
import numpy as np
import model_utils as mu
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

res_path = 'droplet_data_1-0.npz'
in_fpath = './inp_droplets.txt'
out_path = './'
out_prefix = 'raster_data_'

# Simulation parameters
sp = mu.readParams(in_fpath)
dil = sp['Dilution']
t_stop = sp['Ref_tstop']

peak_times = []

results_droplets = np.load(res_path, allow_pickle=True)
for idx, result in enumerate(results_droplets['feature_list']): 
    # Check if that droplet oscillated
    if result is not None:
        print(idx)
        p_droplet = results_droplets['param_list'][idx]
        p_droplet['dil'] = dil 
        # Set Initial condition
        y0 = mu.limit_cycle_init_cond_droplet(int(sp['Num_ODE_Eqs']), p_droplet, 
                                                sp['Gamma_Scale'], sp['Ref_Cyc_B'])
        p_droplet['Init_Cond'] = y0
        # Solve
        sol = solve_ivp(mu.cycle_model, [0, t_stop], y0, first_step = sp['Init_Step_Size'], 
                        max_step = sp['Max_Step_Size'], min_step = sp['Min_Step_Size'], args=([p_droplet]),
                        method = 'LSODA', vectorized = True)
        # I need to obtain peaktimes
        max_idx_pks, _ = find_peaks(sol.y[10, :], distance = 10, prominence = 1e-3)
        time_list = sol.t[max_idx_pks]
        # Save peaktimes
        peak_times.append(time_list)
        break
# Save data for that dilution
np.savez(out_path + out_prefix + str(dil).replace('.','-'), peak_times=peak_times,
         dil=dil)
