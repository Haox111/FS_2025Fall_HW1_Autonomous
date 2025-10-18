import numpy as np
#Predicts 2 number addition for 0 + 0 to 1 + 1
#训练(x1, x2) -> y
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [2]])  # y = x1 + x2

#初始化
np.random.seed(42)
W1 = np.random.randn(2, 2)  #输入层到隐藏层
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)  #隐藏层到输出层
b2 = np.zeros((1, 1))

#超参数
lr = 0.01
epochs = 1000

#定义激活函数
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

#训练循环
for epoch in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = z2

    loss = np.mean((y - y_pred) ** 2)

    dloss_dy_pred = 2 * (y_pred - y) / len(y)
    dW2 = np.dot(a1.T, dloss_dy_pred)
    db2 = np.sum(dloss_dy_pred, axis=0, keepdims=True)
    da1 = np.dot(dloss_dy_pred, W2.T)
    dz1 = da1 * relu_deriv(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

#测试结果
print("\nEstimated Result:")
print(np.hstack((X, y_pred)))
