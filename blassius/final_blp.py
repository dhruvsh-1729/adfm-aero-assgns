import numpy as np
import matplotlib.pyplot as plt

# Define the Blasius equation as a system of first-order ODEs
def blasius_equation(y, eta):
    f, g, h = y
    return [g, h, -0.5 * f * h]

# Implement the fourth-order Runge-Kutta method
def runge_kutta_4(func, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        k1 = np.multiply(dt, func(y[i], t[i]))
        k2 = np.multiply(dt, func(y[i] + 0.5 * k1, t[i] + 0.5 * dt))
        k3 = np.multiply(dt, func(y[i] + 0.5 * k2, t[i] + 0.5 * dt))
        k4 = np.multiply(dt, func(y[i] + k3, t[i] + dt))
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y

# Trapezoidal Rule for Numerical Integration
def trapezoidal_rule(y, h):
    n = len(y)
    integral = 0.5 * (y[0] + y[n - 1])  # Add the first and last terms

    for i in range(1, n - 1):
        integral += y[i]
    
    integral *= h
    return integral

# Set up initial conditions and numerical parameters
eta_max = 10
num_points = 1000
eta_values = np.linspace(0, eta_max, num_points)

p1 = 0.1
p2 = 1.0

y0 = [0, 0, p1]
y1 = [0, 0, p2]

sol1 = runge_kutta_4(blasius_equation, y0, eta_values)
f_values = sol1[:, 0]
g_values = sol1[:, 1]
h_values = sol1[:, 2]

sol2 = runge_kutta_4(blasius_equation, y1, eta_values)
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
    sol = runge_kutta_4(blasius_equation, y, eta_values)
    f2_values = sol[:, 0]
    g2_values = sol[:, 1]
    h2_values = sol[:, 2]
    m2 = g2_values[-1]
    err = abs(m2 - 1)
    count += 1
    print(p2, m2, err, count)

# Calculate boundary layer parameters using the trapezoidal rule
h = eta_values[1] - eta_values[0]
delta = trapezoidal_rule(1 - np.array(g2_values), h)
theta = trapezoidal_rule(np.array(g2_values) * (1 - np.array(g2_values)), h)
shape_factor =delta / theta
Cf = 2*p2

print("Boundary Layer Thickness (delta) coefficient:", delta)
print("Momentum Thickness (theta) coefficient:", theta)
print("Shape Factor (H) coefficient:", shape_factor)
print("Skin Friction Coefficient (Cf):", Cf)

# Plot the results
plt.figure(figsize=(10, 6))
# (Your plotting code here...)
plt.plot(eta_values, f_values, linestyle='dotted', label='f(eta) first guess')
plt.plot(eta_values, g_values, linestyle='dotted', label="f'(eta) first guess")
plt.plot(eta_values, h_values, linestyle='dotted', label="f''(eta) first guess")
plt.plot(eta_values, f1_values, linestyle='dashed', label='f1(eta) second guess')
plt.plot(eta_values, g1_values, linestyle='dashed', label="f1'(eta) second guess")
plt.plot(eta_values, h1_values, linestyle='dashed', label="f1''(eta) second guess")
plt.plot(eta_values, f2_values,label='f1(eta) third guess')
plt.plot(eta_values, g2_values,label="f1'(eta) third guess")
plt.plot(eta_values, h2_values,label="f1''(eta) third guess")
plt.xlabel('eta')
plt.xticks(np.arange(0, 11, 1))
plt.ylabel('sol1, sol2 Values')
plt.yticks(np.arange(0, 11, 1))
plt.legend()
plt.title('Blasius Equation using Runge-Kutta 4 using the shooting method')
plt.grid(True)
plt.show()

