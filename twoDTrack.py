import numpy as np

time_step = 1.0

# State order: [x, y, velocity_x, velocity_y]
position_according_to_velocity = np.array([
    [1, 0, time_step, 0],
    [0, 1, 0, time_step],
    [0, 0, 1,          0],
    [0, 0, 0,          1],
])

# The sensor measures x and y, but not velocity.
position_sensed = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])

# Example sensor readings: (x, y)
measurements = [
    (0.2, -0.1),
    (1.1,  0.4),
    (2.2,  0.8),
    (2.9,  1.3),
    (4.2,  1.7),
]

# Start at the first reading and assume zero initial velocity.
state = np.array([*measurements[0], 0.0, 0.0])

# Initial uncertainty:
# position standard deviation = 1.5 units
# velocity standard deviation = 2.0 units per second
P = np.diag([
    1.5**2,  # x variance
    1.5**2,  # y variance
    2.0**2,  # velocity_x variance
    2.0**2,  # velocity_y variance
])

# Sensor uncertainty:
# position standard deviation = 0.5 units
R = np.diag([
    0.5**2,  # x measurement variance
    0.5**2,  # y measurement variance
])

# Motion uncertainty:
# random acceleration standard deviation = 0.4 units per second squared
acceleration_std = 0.4
acceleration_variance = acceleration_std**2

Q = acceleration_variance * np.array([
    [time_step**4 / 4, 0,                   time_step**3 / 2, 0],
    [0,                   time_step**4 / 4, 0,                   time_step**3 / 2],
    [time_step**3 / 2, 0,                   time_step**2,     0],
    [0,                   time_step**3 / 2, 0,                   time_step**2],
])

for reading in measurements[1:]:
    z = np.array(reading)

    # Predict the next position and uncertainty.
    state_predicted = position_according_to_velocity @ state
    P_predicted = (
        position_according_to_velocity @ P
        @ position_according_to_velocity.T + Q
    )

    # Compare the sensor reading with the predicted position.
    difference = z - position_sensed @ state_predicted
    S = position_sensed @ P_predicted @ position_sensed.T + R

    # Update the estimate using the prediction and sensor reading.
    K = P_predicted @ position_sensed.T @ np.linalg.inv(S)
    state = state_predicted + K @ difference
    P = (np.eye(4) - K @ position_sensed) @ P_predicted

    print(f"sensor:   {z}")
    print(f"estimate: {state[:2]}")
    print(f"velocity: {state[2:]}")
    print()