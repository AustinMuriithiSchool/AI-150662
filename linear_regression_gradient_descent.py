import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Function to compute Mean Squared Error (MSE)
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function for linear regression
def gradient_descent(X, y, m_init=0, c_init=0, learning_rate=0.01, epochs=10):
    m = m_init  # initial slope
    c = c_init  # initial intercept
    n = len(y)  # number of data points
    for epoch in range(epochs):
        # Make predictions
        y_pred = m * X + c
        
        # Compute the gradients
        dm = (-2/n) * np.sum(X * (y - y_pred))  # derivative w.r.t. m
        dc = (-2/n) * np.sum(y - y_pred)        # derivative w.r.t. c
        
        # Update m and c
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Compute the current MSE
        mse = compute_mse(y, y_pred)
        
        # Display the MSE and the current values of m and c for each epoch
        print(f'Epoch {epoch+1}: MSE = {mse:.4f}, m = {m:.4f}, c = {c:.4f}')

    return m, c

# Function to plot the line of best fit
def plot_best_fit(X, y, m, c):
    plt.scatter(X, y, color='blue', label='Data Points')
    
    # Line of best fit
    y_pred = m * X + c
    plt.plot(X, y_pred, color='red', label='Line of Best Fit')
    
    plt.xlabel('Office Size (sq. ft.)')
    plt.ylabel('Office Price')
    plt.title('Line of Best Fit after 10 Epochs')
    plt.legend()
    
    # Save the plot as an image instead of displaying
    plt.savefig('line_of_best_fit.png')
    print("Plot saved as 'line_of_best_fit.png'")

# Load the dataset
file_path = 'Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns: office size (x) and office price (y)
X = data['SIZE'].values
y = data['PRICE'].values

# Run the gradient descent for 10 epochs with random initial values
m_final, c_final = gradient_descent(X, y, m_init=0.5, c_init=1.0, learning_rate=0.0001, epochs=10)

# Plot the final line of best fit
plot_best_fit(X, y, m_final, c_final)

# Predicting the office price for 100 sq. ft.
size_to_predict = 100
predicted_price = m_final * size_to_predict + c_final
print(f"Predicted price for an office of size {size_to_predict} sq. ft. is: {predicted_price:.2f}")
