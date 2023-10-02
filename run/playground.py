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
import custom_tools as ct
import connectivity_creator as conc

# Other modules
import json
from IPython.display import HTML
import matplotlib.animation as animation
from tqdm.autonotebook import trange
import pickle

# Set random seed for consistency
np.random.seed(17)
# Initialize population list

population = []

# Create subplots
fig, axs = plt.subplots(2, 3,figsize=(15,10))



for i in trange(6):

    # Generate a random mechanism
    C,x0,fixed_nodes,motor = random_generator_ns(g_prob = 0.15, n=4, scale=1, strategy='srand')
    C, x0 = ct.append_random_output_node(C, x0, fixed_nodes)
    C, x0, fixed_nodes = ct.append_random_grounded_to_node(C, x0, fixed_nodes, motor[1])
    # C, x0 = ct.append_random_tracer_to_tracers(C, x0, fixed_nodes)
    print(conc.get_list_of_line_nodes(C, fixed_nodes, motor))
    print(conc.get_list_of_loop_nodes(C, fixed_nodes, motor))

    # Plot Mechanism
    draw_mechanism_on_ax(C,x0,fixed_nodes,motor,axs[i//3,i%3])

    # Last node is target
    target = C.shape[0]-1

    # Convert to 1D and add to population
    population.append(to_final_representation(C,x0,fixed_nodes,motor,target))


plt.show()
save_population_csv('../results/0.csv', population)
population_reloaded = get_population_csv('../results/0.csv')
