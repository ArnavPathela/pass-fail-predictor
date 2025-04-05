import numpy as np

import time
# Each column is a student [study_hours, attendance_percent]
X = np.array([
    [2, 60],   # Student 1
    [4, 75],   # Student 2
    [1, 50],   # Student 3
    [3, 90],   # Student 4
    [5, 85],   # Student 5
    [1.5, 40], # Student 6
    [3.5, 70], # Student 7
    [0.5, 30]  # Student 8
]).T  # Shape: (2, 8)

# Labels: 1 = Pass, 0 = Fail
Y = np.array([[0, 1, 0, 1, 1, 0, 1, 0]])  # Shape: (1, 8)

n_x = X.shape[0]
n_h = 16
n_y = Y.shape[0]

W1 = np.random.randn(n_h,n_x)
b1 = np.zeros((n_h,1))
W2 = np.random.randn(n_y,n_h)
b2 = np.zeros((n_y,1))
i = 0
while True:
#forward pass 

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = 1/(1+np.exp(-(Z2)))

    m = X.shape[1]
    # backward propogation 
    dz2 = A2 - Y
    dw2 = (1/m) * np.dot(dz2,A1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims = True)

    da1 = np.dot(W2.T,dz2)

    dz1 = da1 * (1-np.power(A1,2))
    dw1 = (1/m) * np.dot(dz1,X.T)
    db1 = (1/m) * np.sum(dz1 , axis = 1, keepdims= True)


    learning_rate = 0.001

    W1 = W1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    if i % 500 == 0:    
        loss = -np.mean(Y * np.log(A2 + 1e-9) + (1 - Y) * np.log(1 - A2 + 1e-9))
        predictions = (A2 > 0.5).astype(int)
        accuracy = np.mean(predictions == Y)
        print(f"Step {i}, Loss: {loss:.4f}")
        print("Predictions:", predictions)
        print("Actual:", Y)
        print("Accuracy:", accuracy)
        time.sleep(0.2)


