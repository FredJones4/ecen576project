import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a 5x5 matrix of ones
matrix = np.ones((5, 5))
print("Original matrix of ones:")
print(matrix)

# Step 2: Initialize the random number generator
rng = np.random.default_rng()

# Set the average rate (lambda) for the Poisson distribution
lam = 2

# Generate Poisson-distributed noise
poisson_noise = rng.poisson(lam=lam, size=(5, 5))
print("\nPoisson-distributed noise:")
print(poisson_noise)

# Step 3: Add Poisson noise to the original matrix
noisy_matrix = matrix + poisson_noise
print("\nMatrix after adding Poisson noise:")
print(noisy_matrix)

# Step 4: Visualize the Poisson noise distribution
flattened_noise = poisson_noise.flatten()
plt.hist(flattened_noise, bins=range(np.max(flattened_noise) + 2), align='left', edgecolor='black')
plt.title(f'Histogram of Poisson Noise (λ={lam})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 1: Create a 5x5 matrix with varying values
variable_matrix = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9]
])
print("Original variable matrix:")
print(variable_matrix)

# Step 2: Generate Poisson-distributed noise with element-wise lambda
poisson_noise_variable = rng.poisson(lam=variable_matrix)
print("\nElement-wise Poisson-distributed noise:")
print(poisson_noise_variable)

# Step 3: Add Poisson noise to the original variable matrix
noisy_variable_matrix = variable_matrix + poisson_noise_variable
print("\nVariable matrix after adding element-wise Poisson noise:")
print(noisy_variable_matrix)

# Step 4: Visualize the Poisson noise distribution
flattened_noise_variable = poisson_noise_variable.flatten()
plt.hist(flattened_noise_variable, bins=range(np.max(flattened_noise_variable) + 2), align='left', edgecolor='black')
plt.title('Histogram of Element-wise Poisson Noise')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
