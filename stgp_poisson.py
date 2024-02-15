from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from matplotlib import tri
from deap import gp, base
from data.util import load_dataset
import data.poisson.poisson_dataset as pd
from dctkit.mesh import util
from alpine.gp import gpsymbreg as gps
from dctkit import config
import data
import dctkit

import ray

import numpy as np
import jax.numpy as jnp
import math
import time
import sys
import yaml
from typing import Tuple, Callable
import numpy.typing as npt

residual_formulation = False

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()

noise = pd.load_noise()


def is_valid_energy(u: npt.NDArray, prb: oc.OptimizationProblem,
                    bnodes: npt.NDArray) -> bool:
    # perturb solution and check whether the gradient still vanishes
    # (i.e. constant energy)
    u_noise = u + noise*np.mean(u)
    u_noise[bnodes] = u[bnodes]
    grad_u_noise = jnp.linalg.norm(prb.solver.gradient(u_noise))
    is_valid = grad_u_noise >= 1e-6
    return is_valid


def eval_MSE_sol(individual: Callable, indlen: int, X: npt.NDArray,
                 y: npt.NDArray, S: SimplicialComplex, bnodes: npt.NDArray,
                 gamma: float, u_0: C.CochainP0) -> Tuple[float, npt.NDArray]:

    num_nodes = X.shape[1]

    # need to call config again before using JAX in energy evaluations to make
    # sure that  the current worker has initialized JAX
    config()

    # create objective function and set its energy function
    def total_energy(x, curr_y, curr_bvalues):
        penalty = 0.5*gamma*jnp.sum((x[bnodes] - curr_bvalues)**2)
        c = C.CochainP0(S, x)
        fk = C.CochainP0(S, curr_y)
        if residual_formulation:
            total_energy = C.inner_product(individual(c, fk),
                                           individual(c, fk)) + penalty
        else:
            total_energy = individual(c, fk) + penalty
        return total_energy

    prb = oc.OptimizationProblem(
        dim=num_nodes, state_dim=num_nodes, objfun=total_energy)

    MSE = 0.

    us = []

    # Dirichlet boundary conditions for all the samples
    bvalues = X[:, bnodes]

    # loop over dataset samples
    for i, curr_y in enumerate(y):

        curr_bvalues = bvalues[i, :]

        args = {'curr_y': curr_y, 'curr_bvalues': curr_bvalues}
        prb.set_obj_args(args)

        # minimize the objective
        x = prb.solve(x0=u_0.coeffs, ftol_abs=1e-12,
                      ftol_rel=1e-12, maxeval=1000)

        if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                or prb.last_opt_result == 4):
            # check whether the energy is "admissible" (i.e. exclude
            # constant energies)
            valid_energy = is_valid_energy(u=x, prb=prb, bnodes=bnodes)

            if valid_energy:
                current_err = np.linalg.norm(x-X[i, :])**2
            else:
                current_err = math.nan
        else:
            current_err = math.nan

        if math.isnan(current_err):
            MSE = 1e5
            us = [u_0.coeffs]*X.shape[0]
            break

        MSE += current_err

        us.append(x)

    MSE *= 1/X.shape[0]

    return MSE, us


@ray.remote(num_cpus=2)
def eval_best_sols(individual: Callable, indlen: int, X: npt.NDArray,
                   y: npt.NDArray, S: SimplicialComplex, bnodes: npt.NDArray,
                   gamma: float, u_0: npt.NDArray,
                   penalty: dict) -> npt.NDArray:

    _, best_sols = eval_MSE_sol(individual, indlen, X, y, S, bnodes,
                                gamma, u_0)

    return best_sols


@ray.remote(num_cpus=2)
def eval_MSE(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
             S: SimplicialComplex, bnodes: npt.NDArray, gamma: float,
             u_0: npt.NDArray, penalty: dict) -> float:

    MSE, _ = eval_MSE_sol(individual, indlen, X, y, S, bnodes, gamma, u_0)

    return MSE


@ray.remote(num_cpus=2)
def fitness(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
            S: SimplicialComplex, bnodes: npt.NDArray, gamma: float,
            u_0: npt.NDArray, penalty: dict) -> Tuple[float, ]:

    MSE, _ = eval_MSE_sol(individual, indlen, X, y, S, bnodes, gamma, u_0)

    # penalty terms on length
    fitness = MSE + penalty["reg_param"]*indlen

    return fitness,


# Plot best solution
def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, y: npt.NDArray,
             S: SimplicialComplex, bnodes: npt.NDArray,
             gamma: float, u_0: C.CochainP0, toolbox: base.Toolbox,
             triang: tri.Triangulation):

    indfun = toolbox.compile(expr=ind)

    _, u = eval_MSE_sol(indfun, indlen=0, X=X, y=y, S=S, bnodes=bnodes,
                        gamma=gamma, u_0=u_0)

    plt.figure(10, figsize=(8, 4))
    plt.clf()
    fig = plt.gcf()
    _, axes = plt.subplots(2, X.shape[0], num=10)
    for i in range(0, X.shape[0]):
        axes[0, i].tricontourf(triang, u[i], cmap='RdBu', levels=20)
        pltobj = axes[1, i].tricontourf(triang, X[i], cmap='RdBu', levels=20)
        axes[0, i].set_box_aspect(1)
        axes[1, i].set_box_aspect(1)
    plt.colorbar(pltobj, ax=axes)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)


def stgp_poisson(config_file, output_path=None):
    global residual_formulation
    # generate mesh and dataset
    mesh, _ = util.generate_square_mesh(0.08)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    bnodes = mesh.cell_sets_dict["boundary"]["line"]
    num_nodes = S.num_nodes

    np.random.seed(42)
    data_generator_kwargs = {'S': S, 'num_samples_per_source': 4, 'num_sources': 3,
                             'noise': 0.*np.random.rand(num_nodes)}
    data.util.save_datasets(data_path="./",
                            data_generator=pd.generate_dataset,
                            data_generator_kwargs=data_generator_kwargs,
                            perc_val=0.25,
                            perc_test=0.25)

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
        "./", "csv")

    # penalty parameter for the Dirichlet bcs
    gamma = 1000.

    # initial guess for the solution of the Poisson problem
    u_0_vec = np.zeros(num_nodes, dtype=dctkit.float_dtype)
    u_0 = C.CochainP0(S, u_0_vec)

    residual_formulation = config_file["gp"]["residual_formulation"]

    # define primitive set and add primitives and terminals
    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped(
            "MAIN", [C.CochainP0, C.CochainP0], C.Cochain)
        # ones cochain
        pset.addTerminal(C.Cochain(S.num_nodes, True, S, np.ones(
            S.num_nodes, dtype=dctkit.float_dtype)), C.Cochain, name="ones")
    else:
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], float)

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(2., float, name="2.")
    # rename arguments
    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="f")

    penalty = config_file["gp"]["penalty"]
    common_params = {'S': S, 'penalty': penalty, 'bnodes': bnodes,
                     'gamma': gamma, 'u_0': u_0}

    # create symbolic regression problem instance
    gpsr = gps.GPSymbolicRegressor(
        pset=pset, fitness=fitness.remote,
        error_metric=eval_MSE.remote, predict_func=eval_best_sols.remote,
        feature_extractors=[len], print_log=True,
        common_data=common_params, config_file_data=config_file,
        save_best_individual=True, save_train_fit_history=True,
        plot_best=True, plot_best_individual_tree=True,
        output_path="./")

    params_names = ('X', 'y')

    if gpsr.plot_best:
        triang = tri.Triangulation(
            S.node_coords[:, 0], S.node_coords[:, 1], S.S[2])
        gpsr.toolbox.register("plot_best_func", plot_sol, X=X_val, y=y_val,
                              S=S, bnodes=bnodes, gamma=gamma, u_0=u_0,
                              toolbox=gpsr.toolbox, triang=triang)

    start = time.perf_counter()

    # seed = ["SquareF(InnP0(InvMulP0(u, InnP0(u, fk)), delP1(dP0(u))))"]

    gpsr.fit(X_train, y_train, param_names=params_names, X_val=X_val,
             y_val=y_val)

    gpsr.predict(X_test, y_test, param_names=params_names)

    print("Best MSE on the test set: ", gpsr.score(X_test, y_test,
                                                   param_names=params_names))

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))

    # path for output data speficified
    if n_args >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "."

    stgp_poisson(config_file, output_path)
