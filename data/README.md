## Quick explanation on the dataset.
Data files follow .hdf5 types. They have columns - 
Time #  Denotes the time. 
Model # Denotes the model number for which these indices correspond to. Can also be tracer number.
Index # Denotes the Index of the current row. 
(Physical Paramaters) # Set of physical parameters. 
(Species Abundances) # Set of species abundances. 

The datasets were generated using UCLCHEM. The SURFACE and BULK columns are not included because they can be reconstructed from summing the abundances of all surface and bulk species respectively. Electrons are included because they are not conserved (issue which is a work in progress).