PINN Model: "1D_Poi_Cont.py"
--------------------------------------------------
A 5 hidden layer PINN model used to solve the Poission & Continuity equations simultaneously.
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

**All raw data available in "DD_Data.tar.xz"
