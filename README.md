# Power-Noise Verification

### Introduction
This is a Python implementation of **power-noise verification tool** for the chip power grids.

The design proposes a parsing algorithm for power grid design files that can pre-process Spice files to build the data structure of the grid. The design of this data structure facilitates the application of **modified nodal analysis (MNA)**, which transforms the complex abstract grid analysis problem into a simple mathematical problem, a linear system Ax=b. Then the **Cholesky** decomposition, **LU** decomposition of sparse matrix and **conjugate gradient** method are applied to obtain the solution of grid node voltage. 

The simulation results indicate that the tool has high performance and is able to accomplish at least the noise verification for a million-node scale grid.



## Dependencies
- Python >= 3.7
- Python packages: 
  - numpy
  - scipy.sparse
  - [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse) = 0.4.4
- SuiteSparse



## Usage

**INPUT:** the Spice file of power grid design. Some examples of Spice files are saved at `./example/data`. The reference  solution files are also provided in it. Please refer to  [**IBM Power Grid Benchmarks**](https://web.ece.ucsb.edu/~lip/PGBenchmarks/ibmpgbench.html) that our examples come from for more information. 

**OUTPUT:** the solution file for power grid analysis problem, exactly that is the grid node voltage. The solutions of these examples are saved at `./example/output`. The format of solution file is very simple as follows.

`<nodeName>  voltageValue`

