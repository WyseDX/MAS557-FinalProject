Please install the following packages to run the python codes: jax, jaxopt, matplotlib, numpy, time, pickle, os, random, itertools

This code can be run on Google Colab, after installing jaxopt by executing the following:
!pip install —quiet git+https://github.com/google/jaxopt.git


PINN Model: "1D_Poi_Cont.py"
--------------------------------------------------
A 4 hidden layer PINN model used to solve the Poission & Continuity equations simultaneously.
Input layer consists of the positional coordinates x, and applied voltage profile Va.


Refined PINN Model: "1D_Poi_Cont_with_Neumann_BC.py"
--------------------------------------------------
Refined PINN model accounting for potential and charge continuity at source/channel and channel/drain boundaries.


Numerical simulation: "DD_simulation_1D.m"
--------------------------------------------------
Numerical simulation MATLAB code to generate testing and training data. The Scharfetter-Gummel Discretization Scheme is utilized for numerical stability.

Raw data: "Nsd_%.1e-Nch_%.1e-Lch_%d.dat"<br/>
--------------------------------------------------
1st column: Vd<br/>
2nd column: Id<br/>
Next 501 columns: potential profile<br/>
Next 501 columns: charge profile<br/>


Raw data combined: "DD_full_data_Lsd_20.dat"<br/>
--------------------------------------------------
1st column: Nsd<br/>
@@ -15,3 +29,5 @@ Raw data combined: "DD_full_data_Lsd_20.dat"<br/>
5th column: Id<br/>
Next 501 columns: potential profile<br/>
Next 501 columns: charge profile<br/>

**All raw data are available in "DD_Data.tar.xz"


Loss trajectory: "loss_traj_1D_Poi_Cont*.pkl"<br/>
--------------------------------------------------
PINN Model loss trajectory: loss_traj_1D_Poi_Cont.pkl<br/>
Refined PINN Model loss trajectory: loss_traj_1D_Poi_Cont_with_Neumann_BC.pkl<br/>

**All loss trajectory data are available in "loss.tar.xz"


Trained model parameters: "params_1D_Poi_Cont*.pkl"<br/>
--------------------------------------------------
PINN Model trained model parameters: params_1D_Poi_Cont.pkl<br/>
Refined PINN Model trained model parameters: params_1D_Poi_Cont_with_Neumann_BC.pkl<br/>

**All trained model parameters are available in "params.tar.xz"
