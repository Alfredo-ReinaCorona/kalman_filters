import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kalman_output.csv")

plt.figure()

plt.plot(
    df["px_true"],
    df["py_true"],
    label="True Trajectory"
)

plt.scatter(
    df["px_meas"],
    df["py_meas"],
    label="Noisy Measurements",
    s=15
)

plt.plot(
    df["px_est"],
    df["py_est"],
    label="Estimated Trajectory"
)

plt.xlabel("x position")
plt.ylabel("y position")
plt.title("2D Constant-Velocity Kalman Filter")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
