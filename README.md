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

For each benchmark, run the corresponding main script (`stgp_' + benchmark name) and
pass a configuration file (.yaml) as an argument:

```bash
$ python stgp_poisson.py poisson.yaml
```

Check the online documentation of
[_AlpineGP_](https://alpine.readthedocs.io/en/latest/?badge=latest) for the meaning of
the configuration parameters included in the .yaml file.