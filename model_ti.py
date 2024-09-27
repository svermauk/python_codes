# TI of a model potential in NVE ensemble
# By Shivani Verma
# August, 2024 

import numpy as np
import matplotlib.pyplot as plt

# Constants for the MD simulation
mass = 1.0  # mass of the particle
dt = 0.001  # time step
n_steps = 1000000  # number of MD steps
temperature = 1.0  # temperature of the system (k_B T)

# Define potential energy functions U_A(x) and U_B(x)
def U_A(x):
    return 0.5 * x**2

def U_B(x):
    return 0.5 * ((x - 2)**2) + 1

# Define force functions (negative gradient of the potential energy)
def F_A(x):
    return -x  # dU_A/dx = -x

def F_B(x):
    return -(x - 2)  # dU_B/dx = -0.5 * (x - 2)

# Define dU/dλ function
def dU_dl(x, lambda_value):
    return U_B(x) - U_A(x)

# Initialize position and velocity of the particle
x = np.random.normal(0, np.sqrt(temperature / mass))
v = np.random.normal(0, np.sqrt(temperature / mass))

# Thermodynamic Integration (TI) parameters
lambda_values = np.linspace(0, 1, 11)  # 11 lambda points from 0 to 1
average_dU_dl = []  # List to store the ensemble averaged dU/dλ for each lambda value

# Perform TI by running MD simulations at each λ value
for lambda_value in lambda_values:
    dU_dl_accum = 0.0  # Accumulator for dU/dλ
    for step in range(n_steps):
        # Compute the force using the mixed potential U(x, λ) = (1 - λ)U_A(x) + λU_B(x)
        force = (1 - lambda_value) * F_A(x) + lambda_value * F_B(x)
        
        # Velocity Verlet integration
        v += 0.5 * force * dt / mass  # Update velocity (half step)
        x += v * dt  # Update position (full step)
        force = (1 - lambda_value) * F_A(x) + lambda_value * F_B(x)
        v += 0.5 * force * dt / mass  # Update velocity (another half step)
        
        # Accumulate dU/dλ for averaging
        dU_dl_accum += dU_dl(x, lambda_value)
    
    # Ensemble average of dU/dλ for this λ value
    avg_dU_dl = dU_dl_accum / n_steps
    average_dU_dl.append(avg_dU_dl)

# Numerical integration to compute ΔG using the trapezoidal rule
delta_G_values = [np.trapz(average_dU_dl[:i+1], lambda_values[:i+1]) for i in range(len(lambda_values))]

# Writing <dU/dλ>_λ vs λ to dudl.dat
with open("dudl.dat", "w") as f:
    for lam, avg_dU in zip(lambda_values, average_dU_dl):
        f.write(f"{lam}\t{avg_dU}\n")

# Writing free energy (ΔG) vs λ to free_energy.dat
with open("free_energy.dat", "w") as f:
    for lam, delta_G in zip(lambda_values, delta_G_values):
        f.write(f"{lam}\t{delta_G}\n")

# Plotting the <dU/dλ>_λ vs λ curve
plt.plot(lambda_values, average_dU_dl, label='<dU/dλ>_λ', marker='o')
plt.xlabel('λ')
plt.ylabel('<dU/dλ>_λ')
plt.title('Thermodynamic Integration via MD')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the free energy (ΔG) vs λ curve
plt.plot(lambda_values, delta_G_values, label='Free Energy ΔG vs λ', marker='o')
plt.xlabel('λ')
plt.ylabel('Free Energy ΔG')
plt.title('Free Energy as a Function of λ')
plt.legend()
plt.grid(True)
plt.show()

# Output the computed free energy difference at λ=1
print(f"Computed Free Energy Difference ΔG at λ=1: {delta_G_values[-1]:.4f}")

