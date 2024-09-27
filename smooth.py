# Gaussian filter code for smoothening the FES
# By Shivani Verma
# May, 2024

import numpy as np
from scipy.ndimage import gaussian_filter

# Read data from file
with open('fes.dat', 'r') as file:
    lines = file.readlines()

# Extract non-blank lines and parse data
data = []
for line in lines:
    if line.strip():  # Check if line is not blank
        data.append(list(map(float, line.split())))

data = np.array(data)

# Extract x, y, and z coordinates from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Reshape data to 2D grid
n_x = len(np.unique(x))
n_y = len(np.unique(y))
X = x.reshape((n_x, n_y))
Y = y.reshape((n_x, n_y))
Z = z.reshape((n_x, n_y))

# Smooth the surface using a Gaussian filter
sigma = 3.0  # Adjust sigma for the desired smoothing level
Z_smooth = gaussian_filter(Z, sigma)

# Write smoothed data to a new file in the appropriate format for splot
with open('smoothed_fes.dat', 'w') as file:
    for i in range(n_x):
        for j in range(n_y):
            file.write(f"{X[i, j]} {Y[i, j]} {Z_smooth[i, j]}\n")
        file.write('\n')  # Add a blank line after each row

print("Smoothed data written to smoothed_fes.dat")

