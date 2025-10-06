# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Add bias term (x0 = 1) to each instance
X_b = np.c_[np.ones((100, 1)), X]  # shape = (100, 2)

# Gradient Descent Function
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(2, 1)  # initialize weights randomly (2 because we have bias + 1 feature)
    
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
    return theta

# Train the model
theta_best = gradient_descent(X_b, y)

# Output model parameters
print(f"Learned parameters: intercept = {theta_best[0][0]:.4f}, slope = {theta_best[1][0]:.4f}")

# Plotting
plt.scatter(X, y, color='blue', label='Data')
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
plt.plot(X_new, y_predict, color='red', label='Prediction')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
![alt text](<Screenshot 2025-10-07 001416.png>)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
