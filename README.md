This repository contains the scripts to reproduce the benchmark problems discussed in the
paper [_Discovering interpretable physical models with symbolic regression and discrete
exterior calculus_](https://iopscience.iop.org/article/10.1088/2632-2153/ad1af2).

Benchmark problems:
- Poisson equation
- Euler's Elastica
- Linear Elasticity

_Prerequisites_: [AlpineGP](https://github.com/cpml-au/AlpineGP) library along with its
dependencies.

## Running the benchmarks
```bash
§ python stgp_poisson.py poisson.yaml
```