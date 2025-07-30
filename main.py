import numpy as np
import matplotlib.pyplot as plt
from split_data import split_data
from gradient_descent import gradient_descent
from stochastic_gradient import stochastic_gradient
from cosid import cosid

# === READING AND PREPROCESSING DATA ===
file = "Smart_Farming_Crop_Yield_2024.csv"

train_data, test_data = split_data(file)

# === DEFINING VARIABLES FOR GRADIENT DESCENT METHOD ===
m, n = train_data.shape
A_1 = np.hstack((train_data[:, :-1], np.ones((m, 1))))  # Add bias term
training_target = train_data[:, -1].reshape(-1, 1)       # Ensure column vector

learning_rate = 1e-5
max_iter = 10000
gradient_prag = 1e-6
nr_neurons = 30

# === GRADIENT DESCENT METHOD ===
X, x_star, errors, norms, times = gradient_descent(
    A_1, training_target, nr_neurons, learning_rate, max_iter, gradient_prag
)

cumulative_times = np.cumsum(times)

plt.figure(figsize=(10, 16), facecolor='w')
plt.subplot(4, 1, 1)
plt.semilogy(range(1, len(errors) + 1), errors, linewidth=1.5, color='b')
plt.title('Error vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.semilogy(range(1, len(norms) + 1), norms, linewidth=1.5, color='r')
plt.title('Gradient Norm vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Norm')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(cumulative_times, errors, linewidth=1.5, color='g')
plt.title('Error vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(cumulative_times, norms, linewidth=1.5, color='m')
plt.title('Gradient Norm vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Norm')
plt.grid(True)

plt.tight_layout()
plt.show()

# === TESTING ===
A_testing = np.hstack((test_data[:, :-1], np.ones((test_data.shape[0], 1))))
testing_target = test_data[:, -1].reshape(-1, 1)
output = cosid(A_testing @ X) @ x_star

combined_vectors = np.hstack((testing_target, output))
print("Combined vectors (Testing):")
print(combined_vectors)

mse = np.mean((output - testing_target) ** 2)
print(f"Mean Square Error for Gradient Descent Method: {mse}")

# === STOCHASTIC GRADIENT DESCENT METHOD ===
nr_of_examples = 5

X, x_star, errors, norms, times = stochastic_gradient(
    A_1, training_target, nr_neurons, learning_rate, max_iter, gradient_prag, nr_of_examples
)

cumulative_times = np.cumsum(times)

plt.figure(figsize=(10, 16), facecolor='w')
plt.subplot(4, 1, 1)
plt.semilogy(range(1, len(errors) + 1), errors, linewidth=1.5, color='b')
plt.title('Error vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.semilogy(range(1, len(norms) + 1), norms, linewidth=1.5, color='r')
plt.title('Gradient Norm vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Norm')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(cumulative_times, errors, linewidth=1.5, color='g')
plt.title('Error vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(cumulative_times, norms, linewidth=1.5, color='m')
plt.title('Gradient Norm vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Norm')
plt.grid(True)

plt.tight_layout()
plt.show()

# === TESTING ===
output = cosid(A_testing @ X) @ x_star

combined_vectors = np.hstack((testing_target, output))
print("Combined vectors (Testing):")
print(combined_vectors)

mse = np.mean((output - testing_target) ** 2)
print(f"Mean Square Error for Stochastic Gradient Descent Method: {mse}")
