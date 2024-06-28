import numpy as np
import requests
import csv
import tempfile
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/sarwataijaz/Linear-and-Multiple-Regression/main/salary_data.txt"

# Send a GET request to retrieve the data
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
  # Decode the response data (assuming it's UTF-8 encoded)
 # Decode the response data (assuming it's UTF-8 encoded)
  data_string = response.text
    
with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Create temporary file
  temp_file.write(response.content)
  temp_filename = temp_file.name
    
data = np.loadtxt(temp_filename, delimiter=",") # salary (target) and years of experience

X = data[:,0]

Y = data[:,1]

Y = data[:,1].reshape(X.size,1) # x.size number of rows and one column, all stored as separate elements

X = np.vstack((np.ones((X.size, )), X)).T # to generate bias terms and transposing so features column comes first (num of years in this case) 
print(X.shape)
print(Y.shape)

# plot graph to see their linear relation
plt.scatter(X[:,1], Y)
plt.show()

# creating a function which will help us in predicting the values for salaries based on the years

def gradient_descent(X, Y, learning_rate, iterations):

    m = Y.size # total data points
    theta_values = np.zeros((2,1)) # as we have only 2 variables 

    for i in range (iterations):
    
        predicted_value = np.dot(X,theta_values)
        cost = (1/(2*m)) *np.sum(np.square(predicted_value - Y)) # cost function
    
        deriv_theta = (1/m)*np.dot(X.T, predicted_value - Y)
        theta_values = theta_values - learning_rate*(deriv_theta) # will change the values of coefficent so the slope can inc/dec

    return theta_values

iterations = 100
learning_rate = 0.00000005
theta = gradient_descent(X, Y, learning_rate = learning_rate,
iterations = iterations)

# Sample prediction (replace with your desired experience value)
new_experience = 6
new_data = np.array([[1, new_experience]])  # Add bias term for the new data point

# Predict salary for the new experience
predicted_salary = np.dot(new_data, theta)

print("Predicted salary for", new_experience, "years of experience: $", (predicted_salary[0], 2))