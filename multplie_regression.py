import numpy as np
from sklearn.linear_model import LinearRegression

# Create sample data (replace with your actual data)
X = np.array([[5, 5.5, 7], [10000, 25000, 50000], [28, 30, 39], [296, 242, 222]])  # multiple Features like years of experience, salary, age, taxes(independent variables)
y = np.array([34880, 42750 ,43630, 63160])

model = LinearRegression()
model.fit(X, y)

new_data = np.array([[8, 6, 42], [15000, 30000, 60000]])  # Features for new data points

# Predict salaries for the new data points
predicted_salaries = model.predict(new_data)

print("Predicted salaries for the new data points:", predicted_salaries)