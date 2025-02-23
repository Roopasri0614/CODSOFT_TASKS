import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
file_path = r"C:\Users\roopa\OneDrive\Desktop\Sales Prediction\advertising.csv"
df = pd.read_csv(file_path)

# Remove missing values
df.dropna(inplace=True)

# Display basic info
print(df.info())

# Define features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Visualize actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()

# Predict new sales
new_data = pd.DataFrame({'TV': [200], 'Radio': [30], 'Newspaper': [50]})
predicted_sales = model.predict(new_data)
print(f"Predicted Sales: {predicted_sales[0]}")

# Save the model
joblib.dump(model, 'sales_prediction_model.pkl')

