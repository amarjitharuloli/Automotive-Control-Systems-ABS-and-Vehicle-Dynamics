
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# ABS AND VEHICLE DYNAMICS SIMULATION
# ==========================================

# Vehicle parameters
mass = 1500.0          # Vehicle mass (kg)
wheel_radius = 0.30    # Wheel radius (m)
wheel_inertia = 1.8    # Wheel rotational inertia (kg m^2)
gravity = 9.81         # Gravity (m/s^2)

# Braking and tire parameters
brake_torque_max = 1800.0  # Maximum brake torque (N m)
mu_tire = 0.85             # Tire-road friction coefficient
target_slip = 0.18          # Desired ABS slip ratio

# Simulation parameters
initial_speed = 27.78       # 100 km/h in m/s
simulation_time = (0.0, 10.0)
sample_times = np.linspace(0, 10, 2001)

# Simplified tire friction model
def tire_friction(slip):
    # Peak friction near 0.18 slip
    return mu_tire * np.sin(
        1.5 * np.arctan(slip / 0.18)
    )

# State: [vehicle speed, wheel angular speed, distance]
def vehicle_model(t, state, abs_enabled):
    v, omega, distance = state

    # Prevent negative speeds
    v = max(v, 0.0)
    omega = max(omega, 0.0)

    # Slip ratio: 0 = free rolling, 1 = locked wheel
    denominator = max(v, 0.5)
    slip = np.clip(
        (v - wheel_radius * omega) / denominator,
        0.0, 1.0
    )

    # Tire-road friction force
    normal_load = mass * gravity / 4.0
    friction_force = normal_load * tire_friction(slip)

    # Requested brake torque
    if abs_enabled:
        # Simplified proportional ABS controller
        brake_torque = np.clip(
            900.0 + 6000.0 * (target_slip - slip),
            0.0,
            brake_torque_max
        )
    else:
        # Constant brake torque without ABS
        brake_torque = brake_torque_max

    # Wheel rotational dynamics
    wheel_acc = (
        friction_force * wheel_radius - brake_torque
    ) / wheel_inertia

    # Vehicle longitudinal dynamics
    vehicle_acc = -4.0 * friction_force / mass

    # Stop vehicle and wheels from reversing
    if v <= 0 and vehicle_acc < 0:
        vehicle_acc = 0.0

    if omega <= 0 and wheel_acc < 0:
        wheel_acc = 0.0

    return [
        vehicle_acc,
        wheel_acc,
        v
    ]

# Initial wheel speed corresponds to pure rolling
initial_state = [
    initial_speed,
    initial_speed / wheel_radius,
    0.0
]

def simulate(abs_enabled):
    return solve_ivp(
        lambda t, y: vehicle_model(t, y, abs_enabled),
        simulation_time,
        initial_state,
        t_eval=sample_times,
        max_step=0.005,
        rtol=1e-7,
        atol=1e-9
    )

# Run both cases
result_no_abs = simulate(False)
result_abs = simulate(True)

# Extract vehicle speeds
v_no_abs = np.maximum(result_no_abs.y[0], 0)
v_abs = np.maximum(result_abs.y[0], 0)

# Convert speed to km/h
speed_no_abs = v_no_abs * 3.6
speed_abs = v_abs * 3.6

# Extract wheel speeds
wheel_no_abs = result_no_abs.y[1]
wheel_abs = result_abs.y[1]

# Calculate slip ratios
slip_no_abs = np.clip(
    (v_no_abs - wheel_radius * wheel_no_abs)
    / np.maximum(v_no_abs, 0.5), 0, 1
)

slip_abs = np.clip(
    (v_abs - wheel_radius * wheel_abs)
    / np.maximum(v_abs, 0.5), 0, 1
)

# Distance traveled
distance_no_abs = result_no_abs.y[2]
distance_abs = result_abs.y[2]

# Approximate stopping distances at the first stop
def stopping_distance(speed, distance):
    stopped = np.where(speed <= 0.1)[0]
    if len(stopped) > 0:
        return distance[stopped[0]]
    return np.nan

stop_no_abs = stopping_distance(v_no_abs, distance_no_abs)
stop_abs = stopping_distance(v_abs, distance_abs)

# ==========================================
# PLOT RESULTS
# ==========================================

plt.figure(figsize=(11, 9))

plt.subplot(3, 1, 1)
plt.plot(sample_times, speed_no_abs,
         label="Without ABS", color="red")
plt.plot(sample_times, speed_abs,
         label="With ABS", color="blue")
plt.ylabel("Vehicle speed (km/h)")
plt.title("ABS Braking Performance")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(sample_times, slip_no_abs,
         label="Without ABS", color="red")
plt.plot(sample_times, slip_abs,
         label="With ABS", color="blue")
plt.axhline(target_slip, color="black",
            linestyle="--", label="Target slip")
plt.ylabel("Slip ratio")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(sample_times, distance_no_abs,
         label="Without ABS", color="red")
plt.plot(sample_times, distance_abs,
         label="With ABS", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("abs_results.png", dpi=300)
plt.show()

# ==========================================
# RESULTS
# ==========================================

print("\nABS VEHICLE DYNAMICS RESULTS")
print("--------------------------------")

print(f"Initial speed: {initial_speed * 3.6:.1f} km/h")

print(f"Distance traveled in 10 s without ABS: "
      f"{distance_no_abs[-1]:.2f} m")

print(f"Distance traveled in 10 s with ABS: "
      f"{distance_abs[-1]:.2f} m")

print(f"Approximate stopping distance without ABS: "
      f"{stop_no_abs:.2f} m")

print(f"Approximate stopping distance with ABS: "
      f"{stop_abs:.2f} m")
