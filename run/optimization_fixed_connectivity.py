import custom_tools as ct
# Import required libraries
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #Corrects crashing issues in windows

import numpy as np
import matplotlib.pyplot as plt
import pymoo

# pymoo sub sections
# from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize

# Utiliies for working with linkages (DeCoDE Lab)
from linkage_utils import *
import connectivity_creator as conc


# Other modules
import pickle

# Given some fixed connectivity, construct and solve
class mechanism_synthesis_optimization_fixed_con(ElementwiseProblem):
    save_name = "FIXED-CONN"

    # When intializing, set the mechanism size and target curve
    def __init__(self, target_point_cloud, C, fixed_nodes):
        self.C = C
        self.fixed_nodes = fixed_nodes

        N = C.shape[0]
        self.N = N
        variables = dict()

        #Our position matrix consists of Nx2 real numbers (cartesian coordinate values) between 0 and 1
        for i in range(2*N):
            variables["X0" + str(i)] = Real(bounds=(0.0, 1.0))

        # # Our node type vector consists of N boolean variables (fixed vs non-fixed)
        # fixed nodes are predetermined
        # for i in range(N):
        #     variables["fixed_nodes" + str(i)] =  Binary(N)

        # Target is always the last node
        # variables["target"] = Integer(bounds=(1,N-1))

        # Set up some variables in the problem class we inherit for pymoo
        # n_obj=number of objectives, n_constr=number of constraints
        # Our objectives are chamfer distance and material, and they both have constraints.
        super().__init__(vars=variables, n_obj=2, n_constr=2)

        # Store the target curve point cloud
        self.tpc = target_point_cloud


    def convert_1D_to_mech(self, x):
        N = self.N

        # Always last node
        target = N-1

        # Preset
        C = self.C

        # Reshape flattened position matrix to its proper Nx2 shape
        x0 = np.array([x["X0" + str(i)] for i in range(2*N)]).reshape([N,2])

        # preset
        fixed_nodes = self.fixed_nodes

        #We fix the motor and original ground node as 0 and 1 respectively in this implementation
        motor=np.array([0,1])

        return target, C, x0, fixed_nodes, motor

    def _evaluate(self, x, out, *args, **kwargs):
        #Convert to mechanism representation
        target, C, x0, fixed_nodes, motor = self.convert_1D_to_mech(x)

        #Call our evaluate function to get validity, CD, and material use
        valid, CD, mat, _ = evaluate_mechanism(C,x0,fixed_nodes, motor, self.tpc, idx=target, device='cpu',timesteps=2000)

        # check to see if the mechanism is valid
        if not valid:
            # if mechanism is invalid set the objective to infinity
            out["F"] = [np.Inf,np.Inf]
            out["G"] = [np.Inf, np.Inf]
        else:
            out["F"] = [CD,mat]

            # Set constraints as CD<=0.1 and Material<=10
            # Be careful about modifying these - designs that
            # violate the problem constraints will not be scored.
            out["G"] = [CD - 0.1, mat - 10]

    # def convert_mech_to_1D(C,x0,fixed_nodes,motor, target, N):
    #     variables = dict()
    #     for i in range(N):
    #         for j in range(i):
    #             variables["C" + str(j) + "_" + str(i)] = C[i,j]
    #
    #     del variables["C0_1"]
    #
    #     for i in range(2*N):
    #         variables["X0" + str(i)] = x0.flatten()[i]
    #
    #     for i in range(N):
    #         variables["fixed_nodes" + str(i)] = i in fixed_nodes
    #
    #     variables["target"] = target
    #
    #     return variables

if __name__ == "__main__":
    # Initialize an empty list to store target curves
    target_curves = []

    # Loop to read 6 CSV files and store data in target_curves list
    for i in range(6):
        # Load data from each CSV file and append it to the list
        target_curves.append(np.loadtxt('../data/%i.csv' % (i), delimiter=','))

    # Get target curve 0 and plot
    target_index = 3
    run_index = 4
    seed = 50
    np.random.seed(seed)
    target_curve = np.array(target_curves[target_index])

    nl0_targets = [0]
    nl1_targets = [2, 4, 5]
    nl2_targets = [3]

    # Create desired connectivity graph
    C, fixed_nodes, motor = conc.generate_4bar_connectivity()
    # There is now 1 loop (motor) and 1 line
    if target_index in nl0_targets: # 1 NL layer
        print("NL 0 layer")
        # add a loop node
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C = conc.append_node_with_connectivity(C, [loops[-1], lines[-1]])
    elif target_index == 1:
        # output is a line. output from 2 lines
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C, fixed_nodes = conc.append_ground_to_node(C, fixed_nodes, lines[-1])
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C = conc.append_node_with_connectivity(C, [lines[0], lines[-1]]) # line, NL 1
    elif target_index in nl1_targets: # 1 NL layer
        print("NL 1 layer")
        # first add a loop node
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C = conc.append_node_with_connectivity(C, [loops[-1], lines[-1]])
        # add 1 layer of nonlinearity
        # then add another ground node, conn to motor
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C, fixed_nodes = conc.append_ground_to_node(C, fixed_nodes, motor[-1]) # line, NL 1
        # lastly add the loop of NL 1 to new line and loop
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C = conc.append_node_with_connectivity(C, [loops[-1], lines[-1]]) # loop, NL 1
    elif target_index in nl2_targets: # 2 NL layers
        print("NL 2 layer")
        # first add a loop node
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C = conc.append_node_with_connectivity(C, [loops[-1], lines[-1]])
        # add 1 layer of nonlinearity
        # then add another ground node, conn to motor
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C, fixed_nodes = conc.append_ground_to_node(C, fixed_nodes, motor[-1]) # line, NL 1
        # lastly add the loop of NL 1 to new line and loop
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C = conc.append_node_with_connectivity(C, [loops[-1], lines[-1]]) # loop, NL 1
        # add 2nd layer of nonlinearity
        # add another ground node, conn to motor
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C, fixed_nodes = conc.append_ground_to_node(C, fixed_nodes, motor[-1]) # line, NL 1
        # lastly add the loop of NL 2 to new line and loop
        loops = conc.get_list_of_loop_nodes(C, fixed_nodes, motor)
        lines = conc.get_list_of_line_nodes(C, fixed_nodes, motor)
        C = conc.append_node_with_connectivity(C, [loops[-1], lines[-1]]) # loop, NL 2


    # Setup Problem
    problem = mechanism_synthesis_optimization_fixed_con(target_curve, C, fixed_nodes)

    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.mutation.gauss import GaussianMutation
    from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

    # Set up GA with pop size of 100 -- see pymoo docs for more info on these settings!
    algorithm = NSGA2(pop_size=200, sampling=MixedVariableSampling(),
                      mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                      crossover=SBX(eta=0.2, prob=0.9),
                      mutation=GaussianMutation(sigma=1.0),
                      survival=RankAndCrowdingSurvival(),
                      eliminate_duplicates=MixedVariableDuplicateElimination())

    n_gen = 50
    results = minimize(problem,
                       algorithm,
                       ('n_gen', n_gen),
                       verbose=True,
                       save_history=True,
                       seed=seed,
                      )

    #save the results
    save_name = ct.generate_save_name(mechanism_synthesis_optimization_fixed_con.save_name, target_index, run_index)
    with open(save_name, 'wb') as f:
        pickle.dump([results, problem], f)
