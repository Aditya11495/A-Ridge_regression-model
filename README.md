import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.array([[1], [2], [3], [4], [5],[6], [7], [8], [9], [10],[11],[12]]) # feature values
y = np.array([1, 2, 3, 3, 5,6,3,9,8,6,7,8])  # target values



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = Ridge(alpha=1.0)   # alpha = λ (regularization strength)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Model Coefficient (slope):", model.coef_)
print("Model Intercept:", model.intercept_)

new_x = np.array([[19]])
print("Prediction for x=19:", model.predict(new_x))
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Ridge Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ridge Regression')
plt.legend()
plt.show()

