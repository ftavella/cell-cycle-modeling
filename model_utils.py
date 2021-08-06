import copy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def sigmoid(x, a, b):
    """
    Sigmoid function for fitting

    Parameters
    ----------
    x : float or numpy.array
        variable of the function
    a : float
        slope parameter
    b : float
        midpoint parameter

    Returns
    -------
    float or numpy.array
        Sigmoid of X
    """
    ret = 1/(1 + np.exp(-a*(x-b)))
    return ret

def cycle_model(t, y, p):
    """
    ODE model for cell cycle dynamics based on [1] and 
    modified to include a parameter called dil which models
    the effect of changing the cytoplasmic density.

    Parameters
    ----------
    t : float or numpy.array
        Value or array representing time
    y : numpy.array
        Value or array representing concentrations. The model 
        assumes a particular order of variables in y. See first
        line of code for the unpacking of y
    p : dict
        Dictionary containing parameter names as keys and their
        values. See reference_parameters.txt for an example of 
        parameter names and typical values
    
    Returns
    -------
    numpy.array
        Time derivative of the cell cycle model
    
    [1] Zhang, T., Tyson, J. J., & Novák, B. (2013). 
    Role for regulated phosphatase activity in 
    generating mitotic oscillations in Xenopus 
    cell-free extracts. Proceedings of the National 
    Academy of Sciences, 110(51), 20539-20544.
    """
    APCP1, APCP2, APCP3, APCP4, APCP_C20, Cdc25P, CycBT, C20, ENSAPT, GWP, MPF, B55, Wee1 = y
    # Auxiliary calculations
    APC = p['dil']*p['APCT'] - APCP1- APCP2 - APCP3 - APCP4
    H1 = MPF + p['alpha']*(CycBT- MPF)
    vdpAPC = p['kdpAPC_B55']*B55 + p['kdpAPC_bk']
    vdpMPF = p['kdpMPF_25P']*Cdc25P + p['kdpMPF_25'] * (p['dil']*p['Cdc25T'] - Cdc25P)
    vpMPF = p['kpMPF_Wee']*Wee1 + p['kpMPF_WeeP'] * (p['dil']*p['Wee1T'] - Wee1)
    # APC/C Equiations (1-4)
    dAPCP1dt = p['kpAPC']*H1*APC-(vdpAPC+ p['kpAPC']*H1)*APCP1 +vdpAPC*APCP2
    dAPCP2dt = p['kpAPC']*H1*APCP1-(vdpAPC+ p['kpAPC']*H1)*APCP2 +vdpAPC*APCP3
    dAPCP3dt = p['kpAPC']*H1*APCP2-(vdpAPC+ p['kpAPC']*H1)*APCP3 +vdpAPC*APCP4
    dAPCP4dt = p['kpAPC']*H1*APCP3-vdpAPC*APCP4
    # 5
    dAPCP_C20dt = p['kasAC']*(APCP4 - APCP_C20)*(C20 - APCP_C20) \
                  - (p['kdsAC']+vdpAPC+ p['kp20']*H1)*APCP_C20
    # 6
    dCdc25Pdt = p['ka25_H1']*H1*(p['dil']*p['Cdc25T'] -Cdc25P)/(p['J'] + p['dil']*p['Cdc25T'] -Cdc25P) \
                - (p['ki25'] + p['ki25_B55']*B55)*Cdc25P/(p['J'] +Cdc25P)
    # 7
    dCycBTdt = (p['dil']**p['kspow'])*p['ksCyc'] \
               - (p['dil']**p['kdpow'])*(p['kdCyc']+p['kdCyc_APC']*(APCP_C20/(p['J']+CycBT)))*CycBT
    # 8
    dC20dt = p['kdp20'] * (p['dil']*p['C20T']-C20)- p['kp20'] * H1 * C20
    # 9
    dENSAPTdt = p['kpENSA'] * GWP * (p['dil']*p['ENSAT'] - ENSAPT) - p['kdpENSA'] * ENSAPT
    # 10
    dGWPdt = p['kpGW'] * H1 *(p['dil']*p['GWT'] - GWP)/(p['JpGW'] + p['dil']*p['GWT'] - GWP) \
            - (p['kdpGW']+p['kdpGW_B55']*B55)*GWP/(p['JdpGW'] + GWP)
    # 11
    dMPFdt = (p['dil']**p['kspow'])*p['ksCyc'] \
             - (p['dil']**p['kdpow'])*(p['kdCyc']+ p['kdCyc_APC']*(APCP_C20/(p['J']+CycBT)))*MPF \
             - vpMPF * MPF + vdpMPF*(CycBT - MPF)
    # 12
    dB55dt = - p['kasEP'] * (ENSAPT - (p['dil']*p['B55T'] - B55))*B55 \
            + (p['kdsEP']+p['kdpENSA'])*(p['dil']*p['B55T'] - B55)
    # 13
    dWee1dt = (p['kaWee'] + p['kaWee_B55']*B55)*(p['dil']*p['Wee1T'] \
              - Wee1)/(p['J'] + p['dil']*p['Wee1T'] - Wee1) \
              - p['kiWee_H1'] * H1 * Wee1/(p['J']+ Wee1)
    # Multiply by the appropriate time scaling  
    dydt = p['tscale']*np.array([dAPCP1dt, dAPCP2dt, dAPCP3dt, dAPCP4dt, 
                                 dAPCP_C20dt, dCdc25Pdt, dCycBTdt, dC20dt, 
                                 dENSAPTdt, dGWPdt, dMPFdt, dB55dt, dWee1dt])
    return dydt

def cycle_model_all_decays(t, y, p):
    """
    ODE model for cell cycle dynamics based on [1] and 
    modified to include a parameter called dil which models
    the effect of changing the cytoplasmic density. This model
    differs from cycle_model() in that it includes a decay for
    binding, unbinding, phosphorylation, and dephosphorylation. 

    Parameters
    ----------
    t : float or numpy.array
        Value or array representing time
    y : numpy.array
        Value or array representing concentrations. The model 
        assumes a particular order of variables in y. See first
        line of code for the unpacking of y
    p : dict
        Dictionary containing parameter names as keys and their
        values. See reference_parameters.txt for an example of 
        parameter names and typical values
    
    Returns
    -------
    numpy.array
        Time derivative of the cell cycle model
    
    [1] Zhang, T., Tyson, J. J., & Novák, B. (2013). 
    Role for regulated phosphatase activity in 
    generating mitotic oscillations in Xenopus 
    cell-free extracts. Proceedings of the National 
    Academy of Sciences, 110(51), 20539-20544.
    """
    APCP1, APCP2, APCP3, APCP4, APCP_C20, Cdc25P, CycBT, C20, ENSAPT, GWP, MPF, B55, Wee1 = y
    # Auxiliary calculations
    APC = p['dil']*p['APCT'] - APCP1- APCP2 - APCP3 - APCP4
    H1 = MPF + p['alpha']*(CycBT- MPF)
    vdpAPC = (p['kdpAPC_B55']*(p['dil']**p['kdppow']))*B55 + (p['kdpAPC_bk']*(p['dil']**p['kdppow']))
    vdpMPF = (p['kdpMPF_25P']*(p['dil']**p['kdppow']))*Cdc25P + (p['kdpMPF_25']*(p['dil']**p['kdppow'])) * (p['dil']*p['Cdc25T'] - Cdc25P)
    vpMPF = (p['kpMPF_Wee']*(p['dil']**p['kppow']))*Wee1 + (p['kpMPF_WeeP']*(p['dil']**p['kppow'])) * (p['dil']*p['Wee1T'] - Wee1)
    # APC/C Equiations (1-4)
    dAPCP1dt = (p['kpAPC']*(p['dil']**p['kppow']))*H1*APC-(vdpAPC+ (p['kpAPC']*(p['dil']**p['kppow']))*H1)*APCP1 +vdpAPC*APCP2
    dAPCP2dt = (p['kpAPC']*(p['dil']**p['kppow']))*H1*APCP1-(vdpAPC+ (p['kpAPC']*(p['dil']**p['kppow']))*H1)*APCP2 +vdpAPC*APCP3
    dAPCP3dt = (p['kpAPC']*(p['dil']**p['kppow']))*H1*APCP2-(vdpAPC+ (p['kpAPC']*(p['dil']**p['kppow']))*H1)*APCP3 +vdpAPC*APCP4
    dAPCP4dt = (p['kpAPC']*(p['dil']**p['kppow']))*H1*APCP3-vdpAPC*APCP4
    # 5
    dAPCP_C20dt = (p['kasAC']*(p['dil']**p['kaspow']))*(APCP4 - APCP_C20)*(C20 - APCP_C20) \
                  - (p['kdsAC']*(p['dil']**p['kdspow'])+vdpAPC+ (p['kp20']*(p['dil']**p['kppow']))*H1)*APCP_C20
    # 6
    dCdc25Pdt = p['ka25_H1']*(p['dil']**p['kppow'])*H1*(p['dil']*p['Cdc25T'] -Cdc25P)/(p['J'] + p['dil']*p['Cdc25T'] -Cdc25P) \
                - (p['dil']**p['kdppow'])*(p['ki25'] + p['ki25_B55']*B55)*Cdc25P/(p['J'] +Cdc25P)
    # 7
    dCycBTdt = (p['dil']**p['kspow'])*p['ksCyc'] \
               - (p['dil']**p['kdpow'])*(p['kdCyc']+p['kdCyc_APC']*(APCP_C20/(p['J']+CycBT)))*CycBT
    # 8
    dC20dt = (p['kdp20']*(p['dil']**p['kdppow'])) * (p['dil']*p['C20T']-C20)- (p['kp20']*(p['dil']**p['kppow'])) * H1 * C20
    # 9
    dENSAPTdt = (p['kpENSA']*(p['dil']**p['kppow'])) * GWP * (p['dil']*p['ENSAT'] - ENSAPT) - (p['kdpENSA']*(p['dil']**p['kdppow'])) * ENSAPT
    # 10
    dGWPdt = (p['kpGW']*(p['dil']**p['kppow'])) * H1 *(p['dil']*p['GWT'] - GWP)/(p['JpGW'] + p['dil']*p['GWT'] - GWP) \
            - ((p['kdpGW']*(p['dil']**p['kdppow']))+(p['kdpGW_B55']*(p['dil']**p['kdppow']))*B55)*GWP/(p['JdpGW'] + GWP)
    # 11
    dMPFdt = (p['dil']**p['kspow'])*p['ksCyc'] \
             - (p['dil']**p['kdpow'])*(p['kdCyc']+ p['kdCyc_APC']*(APCP_C20/(p['J']+CycBT)))*MPF \
             - vpMPF * MPF + vdpMPF*(CycBT - MPF)
    # 12
    dB55dt = - (p['kasEP']*(p['dil']**p['kaspow'])) * (ENSAPT - (p['dil']*p['B55T'] - B55))*B55 \
            + (p['kdsEP']*(p['dil']**p['kdspow'])+(p['kdpENSA']*(p['dil']**p['kdppow'])))*(p['dil']*p['B55T'] - B55)
    # 13
    dWee1dt = (p['dil']**p['kdppow'])*(p['kaWee'] + p['kaWee_B55']*B55)*(p['dil']*p['Wee1T'] \
              - Wee1)/(p['J'] + p['dil']*p['Wee1T'] - Wee1) \
              - (p['dil']**p['kppow'])*p['kiWee_H1'] * H1 * Wee1/(p['J']+ Wee1)
    # Multiply by the appropriate time scaling  
    dydt = p['tscale']*np.array([dAPCP1dt, dAPCP2dt, dAPCP3dt, dAPCP4dt, 
                                 dAPCP_C20dt, dCdc25Pdt, dCycBTdt, dC20dt, 
                                 dENSAPTdt, dGWPdt, dMPFdt, dB55dt, dWee1dt])
    return dydt

def readParams(fpath):
    """
    Function that reads parameters from file
    and returns a dictionary with the keys as the
    parameter names and their values

    Parameters
    ----------
    fpath : str
        The relative or absolute file path for the
        file containing the parameters
    
    Returns
    -------
    dict
        keys are str parameter names and values are 
        floats
    """
    inp_df = pd.read_csv(fpath, header = None, sep = ' ', 
                         names=['Parameter', 'Value'])
    p = dict(zip(inp_df['Parameter'], inp_df['Value']))
    return p


def assign_droplet_parameters(p_ref, beta):
    """
    Generate a parameter dictionary from a reference. Parameters
    are assigned from a gamma distribution. Synthesis, degradation, 
    and total concentrations are varied. The rest of the parameters
    stay the same as in the reference.

    Parameters
    ----------
    p_ref : dict
        Reference parameter dictionary to be modified. Average values
        are taken from this dict
    beta : float
        Gamma function scale

    Returns
    -------
    dict
        New parameter dictionary with values sampled based on the reference
    """
    p_out = {}
    p_out = copy.deepcopy(p_ref)
    p_out['ksCyc'] = np.random.gamma(p_ref['ksCyc']/beta, scale = beta)
    p_out['kdCyc'] = np.random.gamma(p_ref['kdCyc']/beta, scale = beta)
    p_out['kdCyc_APC'] = np.random.gamma(p_ref['kdCyc_APC']/beta, scale = beta)
    p_out['APCT'] = np.random.gamma(p_ref['APCT']/beta, scale = beta)
    p_out['Cdc25T'] = np.random.gamma(p_ref['Cdc25T']/beta, scale = beta)
    p_out['C20T'] = np.random.gamma(p_ref['C20T']/beta, scale = beta)
    p_out['ENSAT'] = np.random.gamma(p_ref['ENSAT']/beta, scale = beta)
    p_out['GWT'] = np.random.gamma(p_ref['GWT']/beta, scale = beta)
    p_out['B55T'] = np.random.gamma(p_ref['B55T']/beta, scale = beta)
    p_out['Wee1T']= np.random.gamma(p_ref['Wee1T']/beta, scale = beta)
    return p_out


def limit_cycle_init_cond_bulk(dim, p_dict, ref_cyc_B):
    """
    Generate an initial condition that converges to the dominant
    limit cycle for bulk simulations. The initial condition gets
    diluted appropriately based on the parameters of the model.

    Parameters
    ----------
    dim : int
        Number of ODE equations in the model
    p_dict: dict
        Dictionary containing the parameters of the model
    ref_cyc_B : float
        Value for CyclinB to use as initial condition 

    Returns
    -------
    numpy.array
        Initial condition for the ODE system 
    """
    y0 = np.zeros(dim)
    y0[6] = p_dict['dil']*ref_cyc_B
    y0[7] = p_dict['dil']*p_dict['C20T']/2.0
    y0[11] = p_dict['dil']*p_dict['B55T']
    y0[12] = p_dict['dil']*p_dict['Wee1T']
    return y0


def limit_cycle_init_cond_droplet(dim, p_dict, beta, ref_cyc_B):
    """
    Generate an initial condition that converges to the dominant
    limit cycle for droplet simulations.

    Parameters
    ----------
    dim : int
        Number of ODE equations in the model
    p_dict: dict
        Dictionary containing the parameters of the model
    beta : float
        Gamma function scale
    ref_cyc_B : float
        Average value for CyclinB to use in Gamma functoin 

    Returns
    -------
    numpy.array
        Initial condition for the ODE system 
    """
    y0 = limit_cycle_init_cond_bulk(dim, p_dict, ref_cyc_B)
    y0[6] = p_dict['dil']*np.random.gamma(ref_cyc_B/beta, scale = beta)
    return y0


def is_steady_state(sol, p_dict, vel_threshold):
    """
    Check if ODE state is close to a steady state by analyzing the norm
    of the time derivative state-vector and comparing it to a threshold
    value

    Parameters
    ----------
    sol : scipy object as returned by scipy.solve_ivp
    p_dict : dict
        Parameter dictionary
    vel_threshold : float
        Threshold after which the system is considered
        to be in a steady state

    Returns
    -------
    bool
        True if velocity is less than threshold, False otherwise
    """
    ret = False
    last_point = sol.y[:,-1]
    velocity = np.linalg.norm(cycle_model(0, last_point, p_dict))
    if velocity < vel_threshold:
      ret = True
    return ret


def calcPeriod(time, conc, num_peak_thresh):
    """
    Period calculation from single time series using a peak finding algorithm.
    Time series needs to contain more peaks than num_peak_thresh and have a 
    stable amplitude.

    Parameters
    ----------
    time : np.array of floats
        x-axis of time series
    conc : np.array of floats
        y-axis of time series
    num_peak_thresh : int
        Numer of peaks needed to be considered for period calculation

    Returns
    -------
    float
      Period of the time series. Returns NaN if a calculation could
      not be performed of if the number of peaks was less than threshold
    """
    max_idx_pks, _ = find_peaks(conc, distance = 10, prominence = 1e-3)
    if len(max_idx_pks) < num_peak_thresh: 
      # No oscillations
      per = np.NaN
      return [per, max_idx_pks]
    else: 
      # Possible oscillations 
      time_list = time[max_idx_pks]
      conc_list = conc[max_idx_pks]
      # Period
      per_vals = np.diff(time_list)
      per = np.mean(per_vals[1:]) # Don't use first difference
      # Amplitude and Period coefficient of variation
      per_cv = np.std(per_vals[1:])/np.mean(per_vals[1:])
      amp_cv = np.std(conc_list[2:])/np.mean(conc_list[2:]) # Skip possible transitory peaks
      if per < 1e-3 or per_cv > 1e-2 or amp_cv > 1e-2:
        per = np.NaN
        return [per, max_idx_pks]
      else:
        return [per, max_idx_pks]



def calcFeatures(time, conc, num_peak_thresh):
    """
    Amplitude, Period, Rising duration, and Falling duration calculation 
    from single time series using a peak finding algorithm. Time series 
    needs to contain more peaks than num_peak_thresh and have a stable 
    amplitude.

    Parameters
    ----------
    time : np.array of floats
        x-axis of time series
    conc : np.array of floats
        y-axis of time series
    num_peak_thresh : int
        Numer of peaks needed to be considered for period calculation

    Returns
    -------
    dict
      Dictionary containing the features of the system
    """
    feat_dict = {'Amp': np.NaN, 'Per': np.NaN, 'Rise': np.NaN, 'Fall': np.NaN}
    per, max_idx_pks = calcPeriod(time, conc, num_peak_thresh)
    if np.isnan(per):
        return feat_dict
    else:
        # Calculate the rest of the features
        min_idx_pks, _ = find_peaks(-conc, distance = 10, prominence = 1e-3)
        # Oscillation, calculate the rest of the features
        last_times_max = time[max_idx_pks[-2:]]
        last_times_min = time[min_idx_pks[-2:]]
        rise, fall = calcRiseFall(last_times_max, last_times_min)
        amp = conc[max_idx_pks[-1]] - conc[min_idx_pks[-1]]
        feat_dict.update({'Amp': amp, 'Per': per, 'Rise': rise, 'Fall': fall}) 
        return feat_dict


def calcRiseFall(times_max, times_min): 
    """
    Calculate rising and falling period from a list 
    of peak and trough times

    Parameters
    ----------
    times_max : list or numpy.array
        list of peak times
    times_min : list or numpy.array
        list of trough times

    Returns
    -------
    list
        First index contains the rising period. 
        Second one contains the falling one.
    """
    rising = np.NaN
    falling = np.NaN
    if times_max[-1] > times_min[-1]:
        rising = times_max[-1] - times_min[-1]
        falling = times_min[-1] - times_max[-2]
    else:
        falling = times_min[-1] - times_max[-1]
        rising = times_max[-1] - times_min[-2]
    return [rising, falling]
