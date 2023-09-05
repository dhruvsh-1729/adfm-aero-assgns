import numpy as np
import matplotlib.pyplot as plt

# Define the Falkner-Skan equation as a system of first-order ODEs
def falkner_skan_equation(y, eta, m):
    f, g, h = y
    return [g, h, -(m + 0.5) * f * h - m * (1 - g**2)]

# Implement the fourth-order Runge-Kutta method
def runge_kutta_4(func, y0, t, m):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        k1 = np.multiply(dt, func(y[i], t[i], m))
        k2 = np.multiply(dt, func(y[i] + 0.5 * k1, t[i] + 0.5 * dt, m))
        k3 = np.multiply(dt, func(y[i] + 0.5 * k2, t[i] + 0.5 * dt, m))
        k4 = np.multiply(dt, func(y[i] + k3, t[i] + dt, m))
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y

# Set up numerical parameters
eta_max = 10  # Maximum value of the dimensionless variable eta
num_points = 1000  # Number of points for integration
eta_values = np.linspace(0, eta_max, num_points) # Vector for incremental values of eta

# Define the range of parameter m to iterate over
param_range = 0.06
m_values = np.linspace(-param_range, param_range, int(2*param_range / 0.01) + 1)

# Initialize arrays to store results
delta_values = []
theta_values = []
H_values = []
Cf_values = []

# Iterate over different values of m
for m in m_values:
    # Initial guesses for f''(0)
    p1 = 0.1
    p2 = 1.0

    y0 = [0, 0, p1]
    y1 = [0, 0, p2]

    # Perform the numerical integration using Runge-Kutta 4
    sol1 = runge_kutta_4(falkner_skan_equation, y0, eta_values, m)
    sol2 = runge_kutta_4(falkner_skan_equation, y1, eta_values, m)

    # Extract the sol1 components
    f_values = sol1[:, 0]
    g_values = sol1[:, 1]
    h_values = sol1[:, 2]

    # Extract the sol2 components
    f1_values = sol2[:, 0]
    g1_values = sol2[:, 1]
    h1_values = sol2[:, 2]

    err = 1
    err_threshold = 1e-6

    f2_values = []
    g2_values = []
    h2_values = []

    m2 = g1_values[-1]
    m1 = g_values[-1]
    count = 0

    while err > err_threshold:
        p2 = p1 + (p2 - p1) * (1 - m1) / (m2 - m1)

        y = [0, 0, p2]
        sol = runge_kutta_4(falkner_skan_equation, y, eta_values, m)

        f2_values = sol[:, 0]
        g2_values = sol[:, 1]
        h2_values = sol[:, 2]

        m2 = g2_values[-1]

        err = abs(m2-1)
        count += 1
        print(p2,m2,err,count,m)

    # Calculate displacement thickness (delta)
    delta = np.trapz(1 - f2_values, eta_values)

    # Calculate momentum thickness (theta)
    theta = np.trapz(f2_values * (1 - f2_values), eta_values)

    # Calculate shape factor (H)
    H = delta / theta

    # Calculate skin friction coefficient (Cf)
    Cf = 2 * (1 - f2_values[-1])

    delta_values.append(delta)
    theta_values.append(theta)
    H_values.append(H)
    Cf_values.append(Cf)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(m_values, delta_values)
plt.xlabel('Parameter m')
plt.ylabel('Displacement Thickness (delta)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(m_values, theta_values)
plt.xlabel('Parameter m')
plt.ylabel('Momentum Thickness (theta)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(m_values, H_values)
plt.xlabel('Parameter m')
plt.ylabel('Shape Factor (H)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(m_values, Cf_values)
plt.xlabel('Parameter m')
plt.ylabel('Skin Friction Coefficient (Cf)')
plt.grid(True)

plt.tight_layout()
plt.show()
