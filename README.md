# Modeling the effect of cytoplasmic density on the cell cycle
## In vitro cell cycle oscillations exhibit a robust and hysteretic response to changes in cytoplasmic density
### Minjun Jin, Franco Tavella, Shiyuan Wang, Qiong Yang

Bulk simulations in Figure 3B, 3D, 3E, and 3H use the script `simulate_bulk.py`
The script `simulate_bulk.py` takes as input two files: one that specifies the ODE parameters, and another one that specifies the simulation parameters. Also you need to provide a filepath for saving the simulation results (be sure to include a final /), and a desired name for the output. Example files are provided and you can run the script with the code in this repository by typing:
`python simulate_bulk.py ./reference_parameters.txt ./dilution/dil_inp_bulk.txt ./dilution/results_bulk bulk_data`
This script will generate a file called `bulk_data.npz` that contains four main attributes: p_bulk, sim_params, dil_list, and feature_list. The first three are used for bookkeeping, they save the ODE parameters, simulation parameters, and cytoplasmic densities simulated for reference in data analysis. The last attribute, feature_list, contains the observed oscillatory properties for each dilution. It is a list that follows the same order as dil_list, where each entry corresponds to a different cytoplasmic density. Within feature_list, each element is a dictionary that contains the amplitude (`'Amp'`), period (`'Per'`), rising phase (`'Rise'`), and falling phase duration (`'Fall'`).  

The simulations contained in this project were performed using Python 3.7.0
