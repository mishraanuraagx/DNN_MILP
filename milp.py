import torch
import numpy as np
import pulp
from NN import DNN1
# todo make the implementation modular. All different size DNNs similar to this can easily be created and tested against.

# loading the model
model = DNN1()
model.load_state_dict(torch.load('torch_model_state_dict'))
model.eval()

# Initialize the MILP problem
milp_problem = pulp.LpProblem("Adversarial_Example", pulp.LpMinimize)

# Assuming the input is a single flattened image of size 784
x = pulp.LpVariable.dicts("x", range(784), lowBound=0, upBound=1)

# Define the MILP variables for each layer
layers = []
for i, param in enumerate(model.parameters()):
    weight, bias = param.detach().numpy(), next(model.parameters()).detach().numpy()
    layer = pulp.LpVariable.dicts(f"layer_{i}", range(weight.shape[0]))
    for j in range(weight.shape[0]):
        # For each neuron, create a linear combination of inputs and apply ReLU (max(0, x))
        layers[-1].append(pulp.lpSum([weight[j][k] * x[k] for k in range(weight.shape[1])]) + bias[j])
        layer[j] = pulp.LpAffineExpression(layers[-1][j])
        milp_problem += layer[j] >= 0  # ReLU constraint

# Define the objective function (e.g., to maximize the error for a target class)
target_class = 0  # Change as needed
milp_problem += -layers[-1][target_class]  # Negative sign to maximize

# Solve the MILP problem
milp_problem.solve()

# Extract the adversarial example
adv_example = np.array([x[i].varValue for i in range(784)])

# Reshape to the original image format if needed
adv_example = adv_example.reshape(28, 28)