import numpy as np #importing the numerical python library 
import matplotlib.pyplot as plt #importing the matplotlib library for plots

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

# Set up initial conditions and numerical parameters
eta_max = 10  # Maximum value of the dimensionless variable eta
num_points = 1000  # Number of points for integration
eta_values = np.linspace(0, eta_max, num_points) #vector for incremental values of eta

p1=0.1 #first guess for f''(0)
p2 = 1 #second guess for f''(0)

y0 = [0, 0, p1]  #first initial vector
y1 = [0, 0, p2]  #second initial vector

# Perform the numerical integration using Runge-Kutta 4
sol1 = runge_kutta_4(blasius_equation, y0, eta_values)

# Extract the sol1 components
f_values = sol1[:, 0]
g_values = sol1[:, 1]
h_values = sol1[:, 2]

sol2 = runge_kutta_4(blasius_equation, y1, eta_values)

# Extract the sol2 components
f1_values = sol2[:, 0]
g1_values = sol2[:, 1]
h1_values = sol2[:, 2]

err=1 #initializing error value 
err_threshold = 1e-6 #defining the threshold value for error

f2_values = [] #initialization of arrays to store the final results
g2_values = []
h2_values = []

m2 = g1_values[-1] #value of f'(n) where n = 10 or eta_max for the first guess value p1
m1 = g_values[-1] #value of f'(n) where n = 10 or eta_max for the second guess value p2
count=0 #initialization of count to store the count of iterations it takes to converge

while err > err_threshold:
    p2 = p1 + (p2-p1)*(1-m1)/(m2-m1) #newton-raphson method to make the search range smaller

    y = [0,0,p2] #initializing the boundary condition vector using the new p2 value from newton raphson
    sol = runge_kutta_4(blasius_equation, y, eta_values) #computing using RK4

    # Extract the sol components
    f2_values = sol[:, 0]
    g2_values = sol[:, 1]
    h2_values = sol[:, 2]

    m2 = g2_values[-1] #get the value f'(n) where n or eta_max = 10 

    err = abs(m2-1) #update the new error to approach nearer to desired value of f'(eta_max) = 1
    count+=1 #updating count of iterations
    print(p2,m2,err,count) #printing in console the guess value for f''(0), the new f'(n),
    #new error and the count of iterations passed

# Plot the results
plt.figure(figsize=(10, 6))
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
