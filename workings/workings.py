
# Question 2c

import numpy as np
import matplotlib.pyplot as plt

#Create y and Z matrices
y = np.array([0.106, 0.109, 0.217, 0.214, 0.54, 0.52, 1.08, 1.04, 2.09, 2.11, 3.20, 3.13]).reshape(-1, 1)
Z = np.array([5, 5, 10, 10, 25, 25, 50, 50, 100, 100, 150, 150]).reshape(-1, 1)

#Compute Z transpose Z
ZTZ = np.dot(Z.T, Z)

#Solve for coefficients
a = np.dot(np.linalg.inv(ZTZ), np.dot(Z.T, y))

#Compute response of model
y_best = a * Z

#Plotting
plt.plot(Z, y, 'bo', markersize=10)
plt.plot(Z, y_best, 'r', linewidth=3)
plt.title('v_f/g vs m')
plt.ylabel('v_f/g (s)', fontsize=12)
plt.xlabel('m (kg)', fontsize=12)
plt.xlim([0, 160])
plt.ylim([0, 4])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Measured values', 'Line of best fit'])
plt.show()

#Compute mean value of y
y_bar = np.mean(y)

#Compute coefficient of determination
S_t = np.sum((y - y_bar) ** 2)
S_r = np.sum((y - y_best) ** 2)

r2 = (S_t - S_r) / S_t

print("R-squared value:", r2)

# Question 3

#Define parameters
num_iter = 5  # Number of iterations
a = np.zeros((2, num_iter + 1))
e = np.zeros((2, num_iter + 1))
del_a = np.zeros((2, num_iter + 1))
epsilon_a = np.zeros((1, num_iter + 1))

# Actual particle radii
y = np.array([[4, np.sqrt(5)]])

# Initial values
a[:, 0] = np.array([[2.000, 4.000]])   # Assuming initial values for a are known

# Learning rate
h = 1

# Coordinates of the centre of particle 1 and 2
X_1 = [0, 0]
X_2 = [4, 4]

for k in range(5):

    # Distances from the centres of particle 1 and 2
    f1 = ((a[0, k] - X_1[0]) ** 2 + (a[1, k] - X_1[1]) ** 2) ** 0.5
    f2 = ((a[0, k] - X_2[0]) ** 2 + (a[1, k] - X_2[1]) ** 2) ** 0.5

    # Jacobian
    J = np.array([
    [(a[0, k] - X_1[0]) / f1, (a[1, k] - X_1[1]) / f1],  
    [(a[0, k] - X_2[0]) / f2, (a[1, k] - X_2[1]) / f2]   
])


    # Error

    ind1 = y[0,0] - f1
    ind2 = y[0,1] - f2

    e[0,k] = ind1
    e[1,k] = ind2

    #Gradient descent
    #del_a[:,k] = h * -2 * np.dot(J.T , e[:,k])
    
    #Quasi-Newton
    del_a[:,k] = np.dot(np.linalg.inv(np.dot(J.T, J)), np.dot(J.T, e[:,k]))

    #Update a using the computed gradient
    a[:, k + 1] = a[:, k] + del_a[:,k]

    #Compute relative error
    epsilon_a[:,k] = np.sqrt(np.dot(del_a[:,k].T,del_a[:,k])) / np.sqrt(np.dot(a[:, k + 1].T,a[:, k + 1]))



print('residual=', e)
print('coefficients=', a )
print('approx relative error', epsilon_a)


