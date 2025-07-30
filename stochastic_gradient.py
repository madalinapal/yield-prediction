import numpy as np
import time
from cosid import cosid
from cosid_deriv import cosid_deriv

def stochastic_gradient(A, e, m, learning_rate, max_iter, prag_gradient, nr_of_examples):
    lines, columns = A.shape
    X = 0.01 * np.random.randn(columns, m)   # Hidden layer weights
    x_star = 0.01 * np.random.randn(m, 1)    # Output layer weights

    errors = np.zeros(max_iter)
    norms = np.zeros(max_iter)
    times = np.zeros(max_iter)

    gradient_norm = np.inf
    iter = 0

    while iter < max_iter and gradient_norm > prag_gradient:
        start_time = time.time()
        iter += 1

        # Select random batch
        idx = np.random.choice(lines, size=nr_of_examples, replace=False)
        A_batch = A[idx, :]
        e_batch = e[idx].reshape(-1, 1)  # Ensure column vector

        # Forward pass
        hidden_output = cosid(A_batch @ X)             # shape: (batch_size, m)
        predicted_output = hidden_output @ x_star      # shape: (batch_size, 1)
        error = predicted_output - e_batch

        # Gradients
        hidden_output_deriv = cosid_deriv(A_batch @ X)

        dL_dx_star = (hidden_output.T @ error) / nr_of_examples
        dL_dX = (A_batch.T @ (hidden_output_deriv * (error @ x_star.T))) / nr_of_examples

        # Gradient norm
        gradient_vector = np.concatenate([dL_dX.ravel(), dL_dx_star.ravel()])
        gradient_norm = np.linalg.norm(gradient_vector)

        # Update weights
        X -= learning_rate * dL_dX
        x_star -= learning_rate * dL_dx_star

        # Store metrics
        times[iter - 1] = time.time() - start_time
        errors[iter - 1] = np.sum(error ** 2) / 2
        norms[iter - 1] = gradient_norm

        # Optional print
        if iter % 500 == 0:
            print(f"Iteration {iter}: Error = {errors[iter - 1]:.6f}, Gradient Norm = {gradient_norm:.6f}")

    # Trim arrays to actual number of iterations
    errors = errors[:iter]
    norms = norms[:iter]
    times = times[:iter]

    return X, x_star, errors, norms, times
