import sys
import numpy as np
import model_utils as mu
from scipy.integrate import solve_ivp

ref_p_path = './reference_parameters.txt'
in_fpath = './dilution/dil_inp_droplets.txt'

out_path = './dilution/results_droplets/'
out_prefix = 'droplet_data_'
"""
if len(sys.argv) < 2:
    print('Please provide the path to the input file')
else:
    in_fpath = sys.argv[1]
"""
# ODE Reference parameters
p_ref = mu.readParams(ref_p_path)
# Simulation parameters
sp = mu.readParams(in_fpath)
aux_dil_list = np.linspace(int(sp['Dil_Low']), int(sp['Dil_High']), int(sp['Samples_Dil']))
if bool(sp['Reverse_Dil_Order']):
    dil_list = np.flip(aux_dil_list)
else:
    dil_list = aux_dil_list
t_stop = sp['Ref_tstop']
for idx, d in enumerate(dil_list):
    feature_list = np.empty(int(sp['Droplets_Per_Dil']), dtype=dict)
    param_list = np.empty(int(sp['Droplets_Per_Dil']), dtype=dict)
    for jdx, smp in enumerate(range(int(sp['Droplets_Per_Dil']))):
        p_droplet = mu.assign_droplet_parameters(p_ref, sp['Gamma_Scale'])
        p_droplet['dil'] = d 
        # Set Initial condition
        y0 = mu.limit_cycle_init_cond_droplet(int(sp['Num_ODE_Eqs']), p_droplet, 
                                              sp['Gamma_Scale'], sp['Ref_Cyc_B'])
        p_droplet['Init_Cond'] = y0
        param_list[jdx] = p_droplet
        # Solve
        sol = solve_ivp(mu.cycle_model, [0, t_stop], y0, first_step = sp['Init_Step_Size'], 
                        max_step = sp['Max_Step_Size'], min_step = sp['Min_Step_Size'], args=([p_droplet]),
                        method = 'LSODA', vectorized = True)
        # Final velocity
        if not mu.is_steady_state(sol, p_droplet, sp['Vel_Threshold']):
            # Could be an oscillator
            features = mu.calcFeatures(sol.t, sol.y[10, :], sp['Num_of_Peaks'])
            if not np.isnan(features['Per']):
                feature_list[jdx] = features
            else:
                # Could be a non-converged oscillator
                sol_2 = solve_ivp(mu.cycle_model, [sol.t[-1], 4*sol.t[-1]], sol.y[:,-1], first_step = sp['Init_Step_Size'], 
                                max_step = sp['Max_Step_Size'], min_step = sp['Min_Step_Size'], args=([p_droplet]),
                                method = 'LSODA', vectorized = True)
                if not mu.is_steady_state(sol_2, p_droplet, sp['Vel_Threshold']):
                        # Could be an oscillator
                        features_2 = mu.calcFeatures(sol_2.t, sol_2.y[10, :], sp['Num_of_Peaks'])
                        if not np.isnan(features_2['Per']):
                            feature_list[jdx] = features_2
                        else:
                            continue
                else:
                    continue
        else:
            continue   
    # Oscillation Percentage Estimation
    per_list = [d['Per'] for d in feature_list]
    osc_perc = 100*np.count_nonzero(~np.isnan(per_list))/sp['Droplets_Per_Dil']
    # Screen Output
    if osc_perc > 0:
        print(f'Rel. Cyt. Dens.: {d}, <Period>: {np.nanmean(per_list)}, Osc. Perc.: {osc_perc}, Last Time: {sol.t[-1]}', flush=True)
        t_stop = 8*np.nanmean(per_list)
    else:
        print(f'Rel. Cyt. Dens.: {d}, <Period>: None, Osc. Perc.: {osc_perc}, Last Time: {sol.t[-1]}', flush=True)
    # Save data for that dilution
    np.savez(out_path + out_prefix + str(d).replace('.','-'), param_list=param_list,
            feature_list=feature_list, osc_perc = osc_perc, dil=d)
# Save simulation data
np.savez(out_path + 'Simulation_Log', p_ref=p_ref, sim_params=sp, dil_list=dil_list)

