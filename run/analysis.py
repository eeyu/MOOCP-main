import custom_tools as ct
# Import required libraries
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #Corrects crashing issues in windows

# Utiliies for working with linkages (DeCoDE Lab)
from linkage_utils import *

# Other modules
import pickle

from optimization_fixed_connectivity import mechanism_synthesis_optimization_fixed_con

# First load in data
run_index = 4
target_index = 3
problem_name = mechanism_synthesis_optimization_fixed_con.save_name
save_name = ct.generate_save_name(problem_name, target_index, run_index)
with open(save_name, 'rb') as f:  # Python 3: open(..., 'rb')
   results, problem = pickle.load(f)

# Initialize an empty list to store target curves
target_curves = []

# Loop to read 6 CSV files and store data in target_curves list
for i in range(6):
    # Load data from each CSV file and append it to the list
    target_curves.append(np.loadtxt('../data/%i.csv' % (i), delimiter=','))

# Get target curve 0 and plot
target_curve = np.array(target_curves[target_index])

from pymoo.indicators.hv import HV

def plot_HV(F, ref):

    #Plot the designs
    plt.scatter(F[:,1],F[:,0])

    #plot the reference point
    plt.scatter(ref[1],ref[0],color="red")

    #plot labels
    plt.xlabel('Material Use')
    plt.ylabel('Chamfer Distance')

    #sort designs and append reference point
    sorted_performance = F[np.argsort(F[:,1])]
    sorted_performance = np.concatenate([sorted_performance,[ref]])

    #create "ghost points" for inner corners
    inner_corners = np.stack([sorted_performance[:,0], np.roll(sorted_performance[:,1], -1)]).T

    #Interleave designs and ghost points
    final = np.empty((sorted_performance.shape[0]*2, 2))
    final[::2,:] = sorted_performance
    final[1::2,:] = inner_corners

    #Create filled polygon
    plt.fill(final[:,1],final[:,0],color="#008cff",alpha=0.2)

def comparecurves(C, x, fixed_nodes, motor, target, target_pc):
    valid, CD, mat, sol = evaluate_mechanism(C,x,fixed_nodes, motor, target_pc, idx=target,device='cpu',timesteps=2000)
    target_pc = get_oriented(target_pc)
    plt.scatter(sol[:,0],sol[:,1],s=2)
    plt.scatter(target_pc[:,0],target_pc[:,1],s=2)
    plt.title(f"Chamfer Distance: {CD}")
    plt.axis('equal')

if not results.X is None:
    #Specify reference point
    ref_point = np.array([0.1, 10])

    #Calculate Hypervolume
    ind = HV(ref_point)
    hypervolume = ind(results.F)

    #Print and plot
    print('Hyper Volume ~ %f' %(hypervolume))
    plot_HV(results.F, ref_point)
    plt.show()

    if type(results.X)!=dict:
        best_sol = results.X[np.argmin(results.F[:,0])]
    else:
        best_sol = results.X

    target, C, x0, fixed_nodes, motor = problem.convert_1D_to_mech(best_sol)
    draw_mechanism(C, x0, fixed_nodes, motor)
    plt.show()

    comparecurves(C, x0, fixed_nodes, motor, target, target_curve)
    plt.show()

    # first get generalized 1D representation
    mechanisms = []

    for x in results.pop.get("X"):
        target, C, x0, fixed_nodes, motor = problem.convert_1D_to_mech(x)
        mechanisms.append(to_final_representation(C, x0, fixed_nodes, motor, target))
    # Visualize Pareto Front
    visualize_pareto_front(np.array(mechanisms), results.pop.get("F"), target_curve)
    plt.show()
else:
    print('Did Not Find Solutions!!')