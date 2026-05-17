# Multivariate state-space data
# 4D state, 2D measurement

import numpy as np
import matplotlib.pyplot as plt
import random


def generate_4d_2d_data(
    n_steps=100, # Time steps generated
    dt=0.1, # Update speed - updates every 0.1s
    process_std=0.2, # How much disturbance is injected into true dynamics
    measurement_std=1.0, # Deviation in 
    x0=None,
    seed=None, # Different seed = different synthetic dataset
):
    """
    State:
        x = [px, vx, py, vy]^T

    Measurement:
        y = [px, py]^T

    Dynamics:
        x_{k+1} = A x_k + w_k
        y_k = C x_k + v_k
    """

    rng = np.random.default_rng(seed)

    if x0 is None:
        x = np.array([0.0, 1.0, 0.0, 0.5])
    else:
        x = np.array(x0, dtype=float)

    A = np.array([
        [1, dt, 0,  0],
        [0,  1, 0,  0],
        [0,  0, 1, dt],
        [0,  0, 0,  1],
    ])

    C = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])

    Q = process_std**2 * np.array([
        [dt**4 / 4, dt**3 / 2, 0,          0],
        [dt**3 / 2, dt**2,     0,          0],
        [0,          0,        dt**4 / 4, dt**3 / 2],
        [0,          0,        dt**3 / 2, dt**2],
    ])

    R = measurement_std**2 * np.eye(2)

    true_states = []
    measurements = []

    for _ in range(n_steps):
        w = rng.multivariate_normal(
            mean=np.zeros(4),
            cov=Q
        )

        v = rng.multivariate_normal(
            mean=np.zeros(2),
            cov=R
        )

        x = A @ x + w
        y = C @ x + v

        true_states.append(x.copy())
        measurements.append(y.copy())

    return (
        np.array(true_states),
        np.array(measurements),
        A,
        C,
        Q,
        R,
    )


def plot_data(true_states, measurements, A, C, Q, R, estimates):
    px_true = true_states[:, 0]
    vx_true = true_states[:, 1]
    py_true = true_states[:, 2]
    vy_true = true_states[:, 3]

    # Estimates
    px_estimate = estimates[:, 0]
    py_estimate = estimates[:, 2]

    px_meas = measurements[:, 0]
    py_meas = measurements[:, 1]

    fig = plt.figure()
    plt.plot(px_true, py_true, label="True Trajectory")
    plt.scatter(px_meas, py_meas, label="Noisy Measurements", s=15)
    plt.plot(px_estimate, py_estimate, label="Estimated Trajectory")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("2D Constant-Velocity Data")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def kalman_filter(measurements, A, C, Q, R):

    # Initial Estimate
    # x_hat = np.array([0.0,0.0,0.0,0.0]) # State vector 
    # P = np.eye(4) # 4x4 identity matrix (cov(uncertainty) matrix of current state estimate)

    x_hat = np.array([measurements[0, 0], 0.0, measurements[0, 1], 0.0])
    P = 100 * np.eye(4)

    estimates = []

    I = 1*np.eye(4)


    for y in measurements:
        # Step 1 - Prediction - Run for each measurment y=n
        # Extrapolate the state - x_hat_(k+1) = A @ x_hat_k + B @ u_k + w_k
        # There is no control input so it's just x_hat_(k+1) = A @ x_hat_k  + w_k
        x_hat = A @ x_hat

        # Extrapolate the covariance matrix
        # print("A @ P @ np.transpose(A) shape: ", A @ P @ np.transpose(A))
        # print("Q Shape: ", Q)
        P = A @ P @ np.transpose(A) + Q

        # Step 2 - Update
        # Compute the Kalman gain - P_(k^-1) is the P before the update
        # S = uncertainty from projecting your state estimate into measurement space + measurement noise uncertainty (innovation covariance)
        S = C @ P @ np.transpose(C) + R
        K = P @ np.transpose(C) @ np.linalg.inv(S)

        # Update estimate with observation
        # y is the measurment vector
        x_hat = x_hat + K @ (y - C @ x_hat)

        # Update estimate uncertainty
        P = (I - K @ C ) @ P @ np.transpose(I - K@C) + K @ R @ np.transpose(K)

        estimates.append(x_hat.copy())

    return np.array(estimates)


if __name__ == "__main__":
    true_states, measurements, A, C, Q, R = generate_4d_2d_data()
    

    estimates = kalman_filter(measurements, A, C, Q, R)
    # print(estimates)

    plot_data(true_states, measurements, A, C, Q, R, estimates)

    print("true_states shape:", true_states.shape)
    print("measurements shape:", measurements.shape)
