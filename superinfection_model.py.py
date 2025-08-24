import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
beta = 3.0
k = 1000
u = 0.02
v = 0.3
s = 0.5  # superinfection coefficient

e_list = [0.0, 0.33, 0.77]
n = len(e_list)

# Initial conditions
x0 = 99000
y_total = 1000
y0 = [y_total / n] * n  # equally divided
initial_conditions = [x0] + y0

# Time span
t_span = (0, 100)
t_eval = np.linspace(*t_span, 500)

# ODE system
def superinfection_ode(t, vars):
    x = vars[0]
    y = vars[1:]

    dxdt = k - u * x - x * sum(beta * (1 - e_list[j]) * y[j] for j in range(n))
    
    dydt = []
    for i in range(n):
        infection = y[i] * beta * (1 - e_list[i]) * x
        death = -(u + v) * y[i]
        gain_from_lower = s * beta * (1 - e_list[i]) * sum(y[j] for j in range(i))
        loss_to_higher = -s * sum(beta * (1 - e_list[j]) * y[j] for j in range(i + 1, n))
        dyidt = infection + death + gain_from_lower + loss_to_higher
        dydt.append(dyidt)

    return [dxdt] + dydt

# Solve ODE
sol = solve_ivp(superinfection_ode, t_span, initial_conditions, t_eval=t_eval)

# Extract results
x = sol.y[0]
y_vals = sol.y[1:]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sol.t, x, label='Susceptible (x)', color='black')
for i, y_i in enumerate(y_vals):
    plt.plot(sol.t, y_i, label=f'Infected y{i+1} (e={e_list[i]})')

plt.title("Superinfection Dynamics: $x(t)$ and $y_i(t)$")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
