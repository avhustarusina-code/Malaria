import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Parameters
beta = 3.0
k = 1000
u = 0.02

# Create meshgrid for e_i and v
ei_vals = np.linspace(0.0, 1.0, 100)  # Vaccine efficacy from 0 to 1
v_vals = np.linspace(0.01, 1.0, 100)  # Virulence from 0.01 to 1.0
EI, V = np.meshgrid(ei_vals, v_vals)

# Compute R0 over grid
R0_vals = (beta * (1 - EI) / (u + V)) * (k / u)

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(EI, V, R0_vals, cmap='viridis', edgecolor='none')

ax.set_title('$R_0(e_i, v)$ Surface Plot')
ax.set_xlabel('Vaccine Efficacy $e_i$')
ax.set_ylabel('Virulence $v$')
ax.set_zlabel('$R_0$')
fig.colorbar(surf, shrink=0.5, aspect=10, label='$R_0$')

plt.tight_layout()
plt.show()
