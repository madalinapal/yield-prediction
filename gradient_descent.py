import numpy as np
import time
from cosid import cosid
from cosid_deriv import cosid_deriv

def gradient_descent(A, e, m, learning_rate, max_iter, prag_gradient):
    lines, columns = A.shape

    X = 0.01 * np.random.randn(columns, m)  # Hidden layer weights
    x_star = 0.01 * np.random.randn(m, 1)   # Output layer weights

    errors = np.zeros(max_iter)
    norms = np.zeros(max_iter)
    times = np.zeros(max_iter)

    gradient_norm = np.inf
    iter = 0

    while iter < max_iter and gradient_norm > prag_gradient:
        start_time = time.time()
        iter += 1

        # Forward pass
        hidden_output = cosid(A @ X)                     # shape: (lines, m)
        predicted_output = hidden_output @ x_star        # shape: (lines, 1)
        error = predicted_output - e.reshape(-1, 1)      # ensure column vector

        # Backpropagation
        hidden_output_deriv = cosid_deriv(A @ X)         # shape: (lines, m)

        # Gradient computations
        dL_dx_star = (hidden_output.T @ error) / lines   # shape: (m, 1)
        dL_dX = (A.T @ (hidden_output_deriv * (error @ x_star.T))) / lines  # shape: (columns, m)

        # L2 Regularization
        lambda_reg = 0.001
        dL_dX += lambda_reg * X
        dL_dx_star += lambda_reg * x_star

        # Parameter updates
        X -= learning_rate * dL_dX
        x_star -= learning_rate * dL_dx_star

        # Gradient norm
        gradient_vector = np.concatenate([dL_dX.ravel(), dL_dx_star.ravel()])
        gradient_norm = np.linalg.norm(gradient_vector)

        # Performance metrics
        times[iter - 1] = time.time() - start_time
        errors[iter - 1] = np.sum(error ** 2) / (2 * lines)
        norms[iter - 1] = gradient_norm

        # Optional: print progress every 1000 iterations
        if iter % 1000 == 0:
            print(f"Iteration {iter}: Error = {errors[iter - 1]:.6f}, Norm = {gradient_norm:.6f}")

    # Trim unused entries
    errors = errors[:iter]
    norms = norms[:iter]
    times = times[:iter]

    return X, x_star, errors, norms, times
