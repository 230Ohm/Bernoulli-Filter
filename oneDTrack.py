import numpy as np

State = np.array([
    [1.0, 1.0],
    [0.0, 1.0],
])

Pos = np.array([[1.0, 0.0]])  # measures position only
Uncertainty = np.array([
    [0.04, 0.08],
    [0.08, 0.16],
])
Noise = np.array([[4.0]])

x = np.array([[1.0],
              [0.5]])        # column vector: position, velocity
P = np.diag([25.0, 4.0])

measurements = [1.2, 2.1, 2.9, 4.2, 4.8, 6.2]

print("measurement | estimated position | estimated velocity")
print("DEBUG measurements:", measurements)

for measurement in measurements:
    print("DEBUG loop value:", measurement)
    # Predict
    x_pred = State @ x
    P_pred = State @ P @ State.T + Uncertainty

    # Update
    z = np.array([[measurement]])
    innovation = z - Pos @ x_pred
    S = Pos @ P_pred @ Pos.T + Noise
    K = (P_pred @ Pos.T) / S.item()

    x = x_pred + K @ innovation
    P = (np.eye(2) - K @ Pos) @ P_pred

    print(f"{measurement:11.2f} | {x[0, 0]:18.2f} | {x[1, 0]:17.2f}")