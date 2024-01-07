from NN import DNN1
import torch
import numpy as np
import pulp
from pulp import value
import matplotlib.pyplot as plt

weights = []
biases = []
# loading the model
model = DNN1()
model.load_state_dict(torch.load('torch_model_state_dict'))
model.eval()

for layer in model.children():
    if isinstance(layer, torch.nn.Linear):
        weights.append(layer.weight.data.numpy())
        biases.append(layer.bias.data.numpy())

# Extract and set number of layers(K), neurons_per_layer n_k and neurons at output
K = 0
n_k = [784]  # Start with input layer size, assuming it's for MNIST dataset

for layer in model.children():
    if isinstance(layer, torch.nn.Linear):
        K += 1
        n_k.append(layer.out_features)

# Assuming your model's output layer is for a 10-class classification (e.g., digits 0-9)
target_output_index = 6  # If you want to visualize features for the digit

# Now, K is the total number of layers, and n_k is a list of neuron counts per layer


# Define the MILP problem
milp_problem = pulp.LpProblem("Feature_Visualization", pulp.LpMaximize)

# Define bounds for input layer (0-1 for normalized grayscale images)
lb, ub = 0, 1

# Input layer variables (x^0)
x_input = [pulp.LpVariable(f"x_{i}^0", lb, ub) for i in range(784)]  # 784 input pixels
# Parameters from your model
# ... (extract weights and biases from each layer of the model)

# Creating binary variables for ReLU activation (z_j^k)
z = {}
for k in range(0, K):
    z[k] = [pulp.LpVariable(f"z_{j}^{k+1}", cat='Binary') for j in range(n_k[k + 1])]
# Set a large value for M
M = 100
x = None
# Define variables, constraints, and objective for each layer
for k in range(0, K):  # Loop through each layer
    # Layer variables (x^k and s^k)
    x_k = [pulp.LpVariable(f"x_{j}^{k+1}", 0) for j in range(n_k[k + 1])]
    s_k = [pulp.LpVariable(f"s_{j}^{k+1}", 0) for j in range(n_k[k + 1])]

    # Add constraints for each neuron in layer k
    for j in range(n_k[k+1]):
        # Constraint (5a)
        if x == None:
            milp_problem += pulp.lpSum([weights[k][j][i] * x_input[i] for i in range(n_k[k])]) + biases[k][j] == x_k[
                j] - s_k[j]
        else:
            milp_problem += pulp.lpSum([weights[k][j][i] * x[i] for i in range(n_k[k])]) + biases[k][j] == x_k[
                j] - s_k[j]
        # Constraint (5b, 5c) - Introduce binary variables z_j^k for ReLU activation
        milp_problem += x_k[j] >= 0
        milp_problem += s_k[j] >= 0
        milp_problem += x_k[j] <= M * z[k][j]  # Large M, M > max possible value of x[k][j]
        milp_problem += s_k[j] <= M * (1 - z[k][j])
        # objective function at layer k
        # Minimizing activations in intermediate layers
        # if k <= len(n_k)-2:
        #     milp_problem += -1 * x_k[j]

    # Update input layer variables for next iteration
    x = x_k

# Objective for maximizing activation towards target output (e.g., digit '9')
# Objective for targeted activation in the output layer
# for j in range(len(x)):
#     if j == target_output_index - 1:
#         milp_problem += 100 * x[j]  # Maximize activation of target output neuron

# From list of variables and target_output_index is the index for the output to maximize towards
objective_terms = [100 * x[target_output_index - 1]] + [-x[j] for j in range(len(x)) if j != target_output_index - 1]
milp_problem += pulp.lpSum(objective_terms)


# Solve the MILP problem
milp_problem.solve()
# x_input = [f"x_{i}^0" for i in range(784)]
# Extract the optimized pixel values
optimized_pixel_values = [value(var) for var in x_input]

# Visualize the solution
visualized_feature = np.array(optimized_pixel_values).astype(float).reshape(28, 28)
plt.imshow(visualized_feature, cmap='gray')
plt.colorbar()
plt.title("Feature Visualization for Digit " + str(target_output_index))
plt.show()
