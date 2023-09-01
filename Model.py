import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the synthetic dataset
data = pd.read_csv('synthetic_dataset_with_compound_names.csv')

# Extract features (X) and target (y)
X = data.drop(['Curie_Temperature', 'Compound_Name'], axis=1)
y = data['Curie_Temperature']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=['Element1', 'Element2', 'Element3'])

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.3))  # Adjust dropout rate
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))  # Add an additional hidden layer
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                    epochs=150, batch_size=64, verbose=1)

# Save the trained model
model.save('trained_model.h5')

# Example user data
user_data = X_encoded.sample(n=1, random_state=42)  # Replace with your arbitrary user data

# Standardize user data
user_data_scaled = scaler.transform(user_data)

# Predict user data using the trained model
user_pred = model.predict(user_data_scaled)[0][0]

# Plotting Tc with Band Gap
plt.figure(figsize=(12, 6))

# Plot for Band Gap
plt.subplot(1, 2, 1)
plt.scatter(X['Band_Gap'], y, label='Band Gap', color='blue', alpha=0.5)
plt.scatter(user_data['Band_Gap'], user_pred, label='User Data Prediction (Band Gap)', color='red', marker='X', s=100)
plt.xlabel('Band Gap')
plt.ylabel('Curie Temperature')
plt.title('Curie Temperature vs. Band Gap')
plt.legend()

# Plot for Magnetic Moment
plt.subplot(1, 2, 2)
plt.scatter(X['Magnetic_Moment'], y, label='Magnetic Moment', color='green', alpha=0.5)
plt.scatter(user_data['Magnetic_Moment'], user_pred, label='User Data Prediction (Magnetic Moment)', color='orange', marker='X', s=100)
plt.xlabel('Magnetic Moment')
plt.ylabel('Curie Temperature')
plt.title('Curie Temperature vs. Magnetic Moment')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Predicted Curie Temperature for User Data: {user_pred:.2f}")
