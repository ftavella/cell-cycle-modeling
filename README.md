# Modeling the effect of cytoplasmic density on the cell cycle
**Article:** In vitro cell cycle oscillations exhibit a robust and hysteretic response to changes in cytoplasmic density

**Authors:** Minjun Jin<sup>1,2†</sup>, Franco Tavella<sup>1†</sup>, Shiyuan Wang<sup>1</sup>, Qiong Yang<sup>1,2,3*</sup>

<sup>1</sup>Department of Biophysics, University of Michigan, Ann Arbor, Michigan 48109, USA.

<sup>2</sup>Department of Computational Medicine & Bioinformatics, University of Michigan, Ann Arbor, Michigan 48109, USA.

<sup>3</sup>Department of Physics, University of Michigan, Ann Arbor, Michigan 48109, USA. 

<sup>*</sup>Correspondence: Qiong Yang, Email: qiongy@umich.edu

<sup>†</sup>These authors contributed equally to this work

## Bulk simulations in Figure 3B, 3D, 3E, and 3H 
These simulations use the script `simulate_bulk.py`. This script takes two files as input: one that specifies the ODE parameters, and another one that specifies the simulation parameters. Also, you need to provide a filepath for saving the simulation results (be sure to include a final /), and a desired name for the output.

Example files are provided and you can run the script with the code in this repository by typing:

`python simulate_bulk.py ./reference_parameters.txt ./dilution/dil_inp_bulk.txt ./dilution/results_bulk bulk_data`

This script will generate a file called `bulk_data.npz` that contains four main attributes: `p_bulk`, `sim_params`, `dil_list`, and `feature_list`. The first three are used for bookkeeping, they save the ODE parameters, simulation parameters, and cytoplasmic densities simulated for reference in data analysis. The last attribute, `feature_list`, contains the observed oscillatory properties for each dilution. It is a list that follows the same order as `dil_list`, where each entry corresponds to a different cytoplasmic density. Within `feature_list`, each element is a dictionary that contains the amplitude (`'Amp'`), period (`'Per'`), rising phase (`'Rise'`), and falling phase duration (`'Fall'`). Simulations where only total concentrations decay, use the ODE parameter file called `parameters_no_decay.txt`.

## Droplet simulations in Figure 3C, 3D, 3E, 3G, and 3H
 3H 
Droplet simulations use the script `simulate_droplets.py`. Just as bulk simulations, this script receives two files as input for ODE and simulation parameters. The simulation parameters for droplet simulations are different than those of bulk simulations. Within the dilution folder, example files are provided, and the script can be run as:

`python simulate_droplets.py ./reference_parameters.txt ./dilution/dil_inp_droplets.txt ./dilution/results_droplets/ droplet_data_`

The script generates one `.npz` file for each dilution analized. Each `.npz` file contains the informations for all the droplets simulated within that dilution. 

## Raster plot simulation for Figure 3F
Raster plot simulations require peak times to be saved. Thus, a script called `simulate_raster_single_dil.py` was written. This script takes droplet simulation results for a single dilution (obtained with `simulate_droplets.py`) and obtains the peaktimes for droplets that oscillated. 

## Bifurcation data from Figure 3I
The data resulting from the analysis using XPPAUT as well as the ODE file used are stored in the folder `bifurcation/`. The file `result_cdk1_bif_diag.dat` contains the results for the ODE variable CDK.

## Simulations for Figure 3J
These droplet simulations only differ in the initial condition used for the droplets. The concentration of original extract uses the same initial condition as dilution simulations:

![equation](https://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Cbegin%7Balign*%7D%20%26CB%20%3D%200.1%5Ctimes%20d%2C%20%5C%20%5C%20%5C%20C20%20%3D%201.0%5Ctimes%20d%2C%20%5C%5C%20%26B55%3D0.7%5Ctimes%20d%2C%20%5C%20%5C%20%5C%20W%3D1.0%5Ctimes%20d%2C%20%5C%5C%20%26%5Ctext%7Bremaining%20variables%200.0%7D%20%5Cend%7Balign*%7D)

where d represents the cytoplasmic density.

On the other hand, the dilution of concentrated extract uses the long term steady-state solution of the system when it is concentrated 2.0x. The values are stored in the file `conc_steady_state.txt` in the folder `concentration/`.

In both cases, the script `simulate_droplets.py` was used. However, in each case the line specifying the initial condition was modified.

## Supplemental sensitivity simulation
For each parameter in the ODE model we performed a sensitivity analysis on the period vs. dilution curve. The bulk simulations for each parameter and the scripts used are in the folder `sensitivity`.



The simulations contained in this project were performed using Python 3.7.0
