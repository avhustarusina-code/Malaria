import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Adjusted parameters for visible differences
k = 1000         # recruitment rate per year
u = 0.02         # natural death rate per year
v = 0.3          # disease-induced death rate per year
beta = 3.0       # transmission rate per year

efficacies = {
    'No Vaccine (e=0.0)': 0.0,
    'RTS,S/AS01 (e=0.33)': 0.33,
    'R21/Matrix-M (e=0.77)': 0.77
}

x0 = 99000       # susceptible population
y0 = 1000        # infected population

t_span = (0, 20)  # 20 years
t_eval = np.linspace(*t_span, 400)

def model(t, z, e):
    x, y = z
    total = x + y
    dxdt = k - u*x - beta*(1 - e)*x*y / total
    dydt = beta*(1 - e)*x*y / total - y*(u + v)
    return [dxdt, dydt]

fig, ax = plt.subplots(figsize=(12, 6))

for label, e in efficacies.items():
    sol = solve_ivp(model, t_span, [x0, y0], args=(e,), t_eval=t_eval)
    x_t, y_t = sol.y
    # Plot susceptible with solid line
    ax.plot(sol.t, x_t, label=f'Susceptible - {label}')
    # Plot infected with dashed line
    ax.plot(sol.t, y_t, '--', label=f'Infected - {label}')

ax.set_xlabel('Time (years)')
ax.set_ylabel('Population')
ax.set_title('Susceptible and Infected Populations Over Time')
ax.legend(loc='upper right')
ax.grid(True)
plt.show()
