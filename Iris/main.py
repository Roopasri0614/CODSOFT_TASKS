# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Dataset
df = pd.read_csv(r"C:\Users\roopa\OneDrive\Desktop\Iris\IRIS.csv")

# Step 2: Display the First Few Rows
print("First 5 Rows of Dataset:")
print(df.head())

# Step 3: Check Dataset Info
print("\nDataset Information:")
print(df.info())

# Step 4: Check for Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Step 5: Visualizing the Dataset
# Count plot for species distribution
sns.countplot(x=df['species'])
plt.title("Distribution of Species in Dataset")
plt.show()

# Pairplot to visualize feature relationships
sns.pairplot(df, hue="species", diag_kind="kde")
plt.show()

# Step 6: Convert Categorical Target Variable into Numbers
df['species'] = df['species'].astype('category').cat.codes  # 0: setosa, 1: versicolor, 2: virginica

# Step 7: Split Data into Features (X) and Labels (y)
X = df.iloc[:, :-1]  # Features (all columns except species)
y = df.iloc[:, -1]   # Target (species column)

# Step 8: Split Dataset into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Normalize the Data Using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Train the Model using K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 11: Make Predictions on Test Data
y_pred = knn.predict(X_test)

# Step 12: Evaluate the Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 13: Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['setosa', 'versicolor', 'virginica'], 
            yticklabels=['setosa', 'versicolor', 'virginica'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 14: Save the Trained Model and Scaler
joblib.dump(knn, 'iris_knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel and Scaler Saved Successfully!")

# Step 15: Load the Model for Future Predictions
model = joblib.load('iris_knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Step 16: Predict on a New Sample
sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Example input (new flower measurements)
sample_data = scaler.transform(sample_data)  # Apply the same scaling
prediction = model.predict(sample_data)

# Map the prediction to species name
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
print("\nPredicted Species:", species_mapping[prediction[0]])
