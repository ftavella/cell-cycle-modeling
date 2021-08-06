import sys
import numpy as np
import model_utils as mu
from scipy.integrate import solve_ivp

ref_p_path = './reference_parameters.txt'
in_fpath = './dilution/dil_inp_bulk.txt'

out_path = './dilution/results_bulk/'
out_name = 'bulk_data'
if len(sys.argv) < 5:
    print('Please call the script with the following inputs:')
    print(' - Path to file for reference ODE parameters')
    print(' - Path to file for simulation parameters')
    print(' - Path to folder for saving simulation results (include a final /)')
    print(' - Desired filename for the output')
    raise ValueError()
else:
    ref_p_path = sys.argv[1]
    in_fpath = sys.argv[2]
    out_path = sys.argv[3]
    out_name = sys.argv[4]
#
# ODE parameters
p_bulk = mu.readParams(ref_p_path)
# Simulation parameters
sp = mu.readParams(in_fpath)
aux_dil_list = np.linspace(float(sp['Dil_Low']), float(sp['Dil_High']), int(sp['Samples_Dil']))
if bool(sp['Reverse_Dil_Order']):
    dil_list = np.flip(aux_dil_list)
else:
    dil_list = aux_dil_list
t_stop = sp['Ref_tstop']
feature_list = np.empty(int(sp['Samples_Dil']), dtype=dict)
for idx, d in enumerate(dil_list):
    p_bulk['dil'] = d 
    # Set Initial condition
    y0 = mu.limit_cycle_init_cond_bulk(int(sp['Num_ODE_Eqs']), p_bulk, sp['Ref_Cyc_B'])
    p_bulk['Init_Cond'] = y0
    # Solve
    sol = solve_ivp(mu.cycle_model_all_decays, [0, t_stop], y0, first_step = sp['Init_Step_Size'], 
                    max_step = sp['Max_Step_Size'], min_step = sp['Min_Step_Size'], args=([p_bulk]),
                    method = 'LSODA', vectorized = True)
    # Final velocity
    if not mu.is_steady_state(sol, p_bulk, sp['Vel_Threshold']):
        # Could be an oscillator
        features = mu.calcFeatures(sol.t, sol.y[10, :], sp['Num_of_Peaks'])
        if not np.isnan(features['Per']):
            feature_list[idx] = features
        else:
            # Could be a non-converged oscillator
            sol_2 = solve_ivp(mu.cycle_model_all_decays, [sol.t[-1], 5*sol.t[-1]], sol.y[:,-1], first_step = sp['Init_Step_Size'], 
                            max_step = sp['Max_Step_Size'], min_step = sp['Min_Step_Size'], args=([p_bulk]),
                            method = 'LSODA', vectorized = True)
            if not mu.is_steady_state(sol_2, p_bulk, sp['Vel_Threshold']):
                    # Could be an oscillator
                    features_2 = mu.calcFeatures(sol_2.t, sol_2.y[10, :], sp['Num_of_Peaks'])
                    if not np.isnan(features_2['Per']):
                        feature_list[idx] = features_2
                    else:
                        continue
            else:
                continue
    else:
        continue   
    # Screen Output
    if not np.isnan(feature_list[idx]['Per']):
        print(f"Rel. Cyt. Dens.: {d}, Period: {feature_list[idx]['Per']}, Amp: {feature_list[idx]['Amp']}, Last Time: {sol.t[-1]}", flush=True)
        t_stop = 8*feature_list[idx]['Per']
    else:
        print(f"Rel. Cyt. Dens.: {d}, Period: None, Amp: None, Last Time: {sol.t[-1]}", flush=True)
# Save simulation data
np.savez(out_path + out_name, feature_list=feature_list, p_bulk=p_bulk, sim_params=sp, dil_list=dil_list)

