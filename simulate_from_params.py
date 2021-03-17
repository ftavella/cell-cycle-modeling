import sys
import numpy as np
import model_utils as mu
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ref_p_path = './reference_parameters.txt'
in_fpath = './dilution/dil_inp_bulk.txt'

# ODE parameters
p_ODE = mu.readParams(ref_p_path)
# Simulation parameters
sp = mu.readParams(in_fpath)
p_ODE['dil'] = 1.0
# Set Initial condition
y0 = mu.limit_cycle_init_cond_bulk(int(sp['Num_ODE_Eqs']), p_ODE, sp['Ref_Cyc_B'])
p_ODE['Init_Cond'] = y0
t_stop = sp['Ref_tstop'] + 1000
# Solve
sol = solve_ivp(mu.cycle_model, [0, t_stop], y0, first_step = sp['Init_Step_Size'], 
                max_step = sp['Max_Step_Size'], min_step = sp['Min_Step_Size'], args=([p_ODE]),
                method = 'LSODA', vectorized = True)
if not mu.is_steady_state(sol, p_ODE, sp['Vel_Threshold']):
    # Calculate features
    features = mu.calcFeatures(sol.t, sol.y[10, :], sp['Num_of_Peaks'])
# Plot results
species2plot = ['MPF', 'B55', 'CycBT']
color_list = [cm.tab10(x) for x in np.linspace(0,1,len(species2plot))]
s2i = mu.readParams('./species2idx.txt')
fig, axs = plt.subplots(len(species2plot), 1, sharex=True)
for idx,ax in enumerate(axs):
    sp_idx = s2i[species2plot[idx]]
    ax.plot(sol.t, sol.y[sp_idx, :], label=species2plot[idx],
            linewidth=3, color=color_list[idx])
    ax.legend(loc='upper left', fontsize=12)
    ax.tick_params(labelsize=14)
axs[-1].set_xlabel('Time (min)', fontsize=14)
fig.text(0.04, 0.5, 'Relative Concentration (a.u.)', 
         va='center', rotation='vertical', fontsize=14)
plt.show()

