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

# Other modules
import json
from IPython.display import HTML
import matplotlib.animation as animation
from tqdm.autonotebook import trange
import pickle

def add_node_with_connectivity(C, x, new_x: np.array, connection_arr: list):
    # CHange structure to add new node
    expanded_shape = (C.shape[0] + 1, C.shape[1] + 1)
    expanded_C = np.zeros(expanded_shape)
    expanded_C[0:-1, 0:-1] = C

    # connect new node to 2 old nodes
    for i in connection_arr:
        expanded_C[i, -1] = 1
        expanded_C[-1, i] = 1

    expanded_x = np.append(x, [new_x], axis=0)
    return expanded_C, expanded_x

def append_random_output_node(C, x, fixed_nodes):
    # First find the nodes that aren't fixed
    num_nodes = C.shape[0]
    floating_nodes = [i for i in range(num_nodes) if i not in fixed_nodes]

    # connect new node to 2 old nodes
    base_nodes = np.random.choice(floating_nodes, size=2, replace=False)
    new_x = [np.random.random(), np.random.random()]

    return add_node_with_connectivity(C, x, new_x, base_nodes)

# Is output node if all connected nodes are floating nodes
def check_if_output_node(C, fixed_nodes, node):
    if node in fixed_nodes:
        return False

    num_nodes = C.shape[0]
    floating_nodes = [i for i in range(num_nodes) if i not in fixed_nodes]

    connected_nodes = np.array(list(range(num_nodes)))[C[node, :]==1]

    for i in connected_nodes:
        if i not in floating_nodes:
            return False
    return True

def get_list_of_output_nodes(C, fixed_nodes):
    output_nodes = []
    num_nodes = C.shape[0]
    for i in range(num_nodes):
        if check_if_output_node(C, fixed_nodes, i):
            output_nodes.append(i)
    return np.array(output_nodes)

def append_random_grounded_to_node(C, x, fixed_nodes, f_old):
    num_nodes = C.shape[0]
    # append f1 to old floating
    connected_node = [f_old]
    new_x = [np.random.random(), np.random.random()]
    C, x = add_node_with_connectivity(C, x, new_x, connected_node)
    f1 = num_nodes

    # append ground to f1
    connected_node = [f1]
    new_x = [np.random.random(), np.random.random()]
    C, x = add_node_with_connectivity(C, x, new_x, connected_node)
    ground = num_nodes + 1

    # # append new output to f_old, f1
    connected_node = [f1, f_old]
    new_x = [np.random.random(), np.random.random()]
    C, x = add_node_with_connectivity(C, x, new_x, connected_node)

    fixed_nodes = np.append(fixed_nodes, ground)

    return C, x, fixed_nodes

# Add a motor to increase DOF by 1.
# adds 3 nodes: 1 floating, 1 ground, 1 output
# connection: [ground, f1], [f1, f_old], [output, f1], [output, f_old]
def append_random_1dof_node_to_output(C, x, fixed_nodes):
    # First find the nodes that aren't fixed
    num_nodes = C.shape[0]
    floating_nodes = [i for i in range(num_nodes) if i not in fixed_nodes]

    # look for output_nodes
    output_nodes = get_list_of_output_nodes(C, fixed_nodes)

    # append f1 to old floating
    f_old = np.random.choice(output_nodes, size=1, replace=False)[0]

    return append_random_grounded_to_node(C, x, fixed_nodes, f_old)

def append_random_tracer_to_tracers(C, x, fixed_nodes):
    # First find the nodes that aren't fixed
    num_nodes = C.shape[0]
    floating_nodes = [i for i in range(num_nodes) if i not in fixed_nodes]

    # look for output_nodes
    output_nodes = get_list_of_output_nodes(C, fixed_nodes)

    # append f1 to old floating
    f_old = np.random.choice(floating_nodes, size=2, replace=False)
    # # append new output to f_old, f1
    connected_node = f_old
    print(connected_node)
    new_x = [np.random.random(), np.random.random()]
    C, x = add_node_with_connectivity(C, x, new_x, connected_node)
    return C, x


def generate_save_name(optimizer_name, target_curve_index, run_index):
    return "./" + optimizer_name + "_" + str(target_curve_index) + "_" + str(run_index) + ".pkl"