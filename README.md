# Nonnegative Tensor Completion via Integer Optimization
This repository is an realization of the algorithm proposed in this [paper](https://arxiv.org/abs/2111.04580).

## DEPENDENCIES

This code is written in Python 3. It requires:
 
1. Gurobi Optimizer and gurobipy package (https://www.gurobi.com/), 
   which have a free Academic License
2. PyTen package (https://github.com/datamllab/pyten)
3. Standard packages like Numpy, Scipy, etc.

## INSTALLATION

1. Install Gurobi and gurobipy
(https://www.gurobi.com/documentation/9.5/quickstart_windows/software_installation_guid.html)

2. Install PyTen package: The github code for PyTen is for Python 2. This zip archive contains 
   the file "pyten-master-v3.zip". To install PyTen for Python 3, decompress this zip archive
   and run the command "python setup.py install" in the "pyten-master" folder.

3. Install any needed missing standard packages.

## RUNNING AN EXPERIMENT

To run an experiment, use the command "python runexp.py". To change the problem instance, edit 
the file "runexp.py" in the section of the code labeled "EDIT THESE TO CHANGE THE PROBLEM SETUP".
The problem instance is dicated by the variables:

1. "r" is a list of the tensor dimensions. A tensor with dimensions 10 x 20 x 30 x 40 would be 
   specified with "r = (10,20,30,40)".

2. "n" is the number of samples (with repetition).

3. "corners" is the number of vertices from the norm-1 nonnegative tensor ball used to construct 
   the "true tensor" using the procedure described in the manuscript.

4. "reps" is the number of times to repeat a single experiment.