# Import required libraries
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #Corrects crashing issues in windows

# Utiliies for working with linkages (DeCoDE Lab)
from linkage_utils import *


# Start with 4-bar. this has 1 line, 1 loop
def generate_4bar_connectivity():
    C, x0, fixed_nodes, motor = random_generator_ns(g_prob=0.15, n=4, scale=1, strategy='srand')
    return C, fixed_nodes, motor

def append_node_with_connectivity(C, connections: list):
    # CHange structure to add new node
    expanded_shape = (C.shape[0] + 1, C.shape[1] + 1)
    expanded_C = np.zeros(expanded_shape)
    expanded_C[0:-1, 0:-1] = C

    # connect new node to other nodes
    for i in connections:
        expanded_C[i, -1] = 1
        expanded_C[-1, i] = 1

    return expanded_C

# connect ground to some node
def append_ground_to_node(C, fixed_nodes, node):
    num_nodes = C.shape[0]
    # append f1 to old floating
    connected_nodes = [node]
    C = append_node_with_connectivity(C, connected_nodes)
    f1 = num_nodes

    # append ground to f1
    connected_nodes = [f1]
    C = append_node_with_connectivity(C, connected_nodes)
    ground = num_nodes + 1

    # # # append new output to f_old, f1
    # connected_nodes = [f1, node]
    # C = append_node_with_connectivity(C, connected_nodes)

    fixed_nodes = np.append(fixed_nodes, ground)

    return C, fixed_nodes

# Node traces some line
def check_if_line_node(C, fixed_nodes, motor, node):
    # is a line node if connected to A. at least 1 ground (non-motor) node or B. connected to 2 line nodes
    # B. might be inaccurate
    if node in fixed_nodes:
        return False
    if node in motor:
        return False

    num_nodes = C.shape[0]

    connected_nodes = np.array(list(range(num_nodes)))[C[node, :]==1]

    # connected to at least 1 fixed
    for i in connected_nodes:
        if i in fixed_nodes:
            return True

    # if all floating, check that all nodes are lines
    for i in connected_nodes:
        if not check_if_line_node(C, fixed_nodes, motor, i):
            return False
    return True

# Node traces a complete loop
def check_if_loop_node(C, fixed_nodes, motor, node):
    # is a loop if A. is the motor-connected node and B. if connected to no fixed nodes AND at least 1 loop node
    if node in fixed_nodes:
        return False
    if node == motor[1]:
        return True

    num_nodes = C.shape[0]
    connected_nodes = np.array(list(range(num_nodes)))[C[node, :]==1]

    # check no fixed connections
    for i in connected_nodes:
        if i in fixed_nodes:
            return False

    # check if at least 1 loop node
    for i in connected_nodes:
        if check_if_loop_node(C, fixed_nodes, motor, i):
            return True
    return False

def get_list_of_line_nodes(C, fixed_nodes, motor):
    output_nodes = []
    num_nodes = C.shape[0]
    for i in range(num_nodes):
        if check_if_line_node(C, fixed_nodes, motor, i):
            output_nodes.append(i)
    return np.array(output_nodes)

def get_list_of_loop_nodes(C, fixed_nodes, motor):
    output_nodes = []
    num_nodes = C.shape[0]
    for i in range(num_nodes):
        if check_if_loop_node(C, fixed_nodes, motor, i):
            output_nodes.append(i)
    return np.array(output_nodes)