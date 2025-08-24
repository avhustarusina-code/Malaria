import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 3.0
k = 1000
u = 0.02
v_values = np.linspace(0.01, 1.0, 300)  # Range of virulence values
efficacies = [0.0, 0.33, 0.77]  # Vaccine efficacies
colors = ['blue', 'orange', 'green']

# R0 function
def R0(beta, k, u, v, ei):
    return (beta * (1 - ei) / (u + v)) * (k / u)

# Plot
plt.figure(figsize=(9, 6))

for ei, color in zip(efficacies, colors):
    R0_vals = R0(beta, k, u, v_values, ei)
    plt.plot(v_values, R0_vals, label=f"$e_i$ = {ei}", color=color)

plt.title("$R_0(v)$ for different vaccine efficacies")
plt.xlabel("Virulence $v$")
plt.ylabel("$R_0(v)$")
plt.axhline(1, color='red', linestyle='--', label="$R_0 = 1$ threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
