import pandas as pd
import numpy as np
import tensorflow as tf
import folium
from folium import Circle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
# Path to your CSV file (update the path to the actual location of your file)
csv_file_path = 'EQ Data.csv'

# Load the dataset
earthquake_data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataset
print(earthquake_data.head())

# Convert 'time' to a datetime format (handling potential errors in parsing)
earthquake_data['time'] = pd.to_datetime(earthquake_data['time'], errors='coerce')

# Check if the dataset has been loaded properly, displaying the data types
print(earthquake_data.dtypes)

# Display a summary of numeric columns
print(earthquake_data.describe())

# Display information about the dataset to check for null values and data types
print(earthquake_data.info())

# Show the unique values in 'magType' to understand different magnitude types
print("Unique magnitude types:", earthquake_data['magType'].unique())

# Check how many missing values are present in each column
print("Missing values per column:")
print(earthquake_data.isnull().sum())

# Handling missing data: You can choose to fill missing values, drop rows, or other strategies.
# Example: Filling missing numerical values with the median of the column
earthquake_data.fillna({
    'depth': earthquake_data['depth'].median(),
    'mag': earthquake_data['mag'].median(),
    'magNst': earthquake_data['magNst'].median(),
    'horizontalError': earthquake_data['horizontalError'].median(),
    'depthError': earthquake_data['depthError'].median(),
    'magError': earthquake_data['magError'].median(),
    'nst': earthquake_data['nst'].median(),      # Filling missing values in 'nst'
    'gap': earthquake_data['gap'].median(),      # Filling missing values in 'gap'
    'dmin': earthquake_data['dmin'].median()  
}, inplace=True)

# Check again to ensure no missing data remains
print("Missing values after cleaning:")
print(earthquake_data.isnull().sum())

# Now let's focus on the core features we will use for prediction or analysis
# Example of feature selection: Latitude, Longitude, Depth, Magnitude (mag), and others
selected_features = earthquake_data[['time', 'latitude', 'longitude', 'depth', 'mag', 'magType', 'place']]

# Display the selected features
print("Selected features:")
print(selected_features.head())

# Saving cleaned data to a new CSV file (if needed)
cleaned_file_path = 'cleaned_earthquake_data.xlsx'
earthquake_data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to: {cleaned_file_path}")

earthquake_data = pd.read_csv(cleaned_file_path)

# Display the first few rows to ensure data is loaded correctly
print(earthquake_data.head())

def categorize_magnitude(mag):
    if mag > 5.0:
        return 'Significant'
    elif mag >= 4.0:
        return 'Moderately Significant'
    else:
        return 'Less Significant'

# Apply the categorization
earthquake_data['magnitude_category'] = earthquake_data['mag'].apply(categorize_magnitude)

# Check for any missing values
print("\nMissing Values Check:")
print(earthquake_data.isnull().sum())

# Fill missing values with median or other appropriate methods
earthquake_data.fillna({
    'nst': earthquake_data['nst'].median(),
    'gap': earthquake_data['gap'].median(),
    'dmin': earthquake_data['dmin'].median()
}, inplace=True)

# Features and target selection
features = earthquake_data[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'horizontalError', 'depthError', 'magError']]
target = earthquake_data['magnitude_category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['Significant', 'Moderately Significant', 'Less Significant']))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=['Significant', 'Moderately Significant', 'Less Significant']))

# Check the distribution of actual and predicted values
print("\nActual vs Predicted Distribution:")
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).value_counts())
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Save the predictions to an Excel file
prediction_output_path = 'earthquake_predictions.xlsx'
output_df = X_test.copy()
output_df['Actual'] = y_test.values
output_df['Predicted'] = y_pred

output_df.to_excel(prediction_output_path, index=False)

print(f"Predictions saved to: {prediction_output_path}")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Setup the GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Use the best model
best_rf_model = grid_search.best_estimator_
# Encode target variable
target_encoded = pd.get_dummies(target, drop_first=True)  # One-hot encode

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5), 
    Dense(64, activation='relu'),
    Dropout(0.5), 
    Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)


# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Neural Network Accuracy: {accuracy:.2f}")

# Make predictions
y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)  # Get the class with the highest probability

# Convert predictions to categorical format
y_test_labels = y_test.values.argmax(axis=1)
cm = confusion_matrix(y_test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Print confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_labels, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, labels=['Significant', 'Moderately Significant', 'Less Significant']))
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
  #-------------------------MAGNITUDE PREDICTION------------------------------------
features = earthquake_data[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms']]
target = earthquake_data['mag']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest Regression - MAE: {mae:.2f}, MSE: {mse:.2f}")

magnitude_output_path = 'magnitude_predictions.xlsx'
output_df = X_test.copy()
output_df['Actual'] = y_test.values
output_df['Predicted'] = y_pred

output_df.to_excel(magnitude_output_path, index=False)

print(f"Predictions saved to: {magnitude_output_path}")
nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(1)  # Output layer for regression
])

# Compile the model
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# Train the model
history = nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

# Predict on test set
y_pred_nn = nn_model.predict(X_test)

# Evaluate the model
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f"Neural Network Regression - MAE: {mae_nn:.2f}, MSE: {mse_nn:.2f}")
print(f"Random Forest Regression - MAE: {mae:.2f}, MSE: {mse:.2f}")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Random Forest Regression')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')

# Plot predictions vs actual values for Neural Network
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.title('Neural Network Regression')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')

plt.tight_layout()
plt.show()
def create_map(data):
    
    
    # Initialize a folium map centered on a location
    map_center = [data['latitude'].mean(), data['longitude'].mean()]
    map = folium.Map(location=map_center, zoom_start=5)
    
    # Iterate over the data and add circles to the map for each earthquake
    for _, row in data.iterrows():
        if 'mag' in row:
            Circle(
                location=[row['latitude'], row['longitude']],
                radius=row['mag'] * 2000,  # Adjust magnitude for better visibility
                color='red',
                fill=True,
                fill_opacity=0.6
            ).add_to(map)
        else:
            print(f"Missing 'mag' value for row: {row}")

    return map

# Generate and display the map
earthquake_map = create_map(earthquake_data)
earthquake_map.save("earthquake_map.html")

st.title("Real-Time Earthquake Management System")
st.write("Real-time Earthquake Data", features)

# Visualize on map
st.subheader("Earthquake Map")
st.write("View the location of recent earthquakes.")
map = create_map(features)
st.components.v1.html(map._repr_html_(), height=500)

# Model Prediction
st.subheader("Prediction")
st.write("Predict earthquake magnitude using your trained model:")
predicted_magnitudes = model.predict(X_test)

# Add the predictions to the DataFrame
# Downloadable Report
if st.button("Save Predictions"):
        features_copy = features.copy()
        features_copy.loc[X_test.index, 'Predicted Magnitude'] = predicted_magnitudes
        prediction_output_path = 'earthquake_pred.xlsx'
        features_copy.to_excel(prediction_output_path, index=False)
        st.write(f"Predictions saved to: {prediction_output_path}")


def generate_emergency_recommendations(magnitude,latitude,longitude):
    recommendations = []
    
    # Example recommendations based on magnitude
    if magnitude > 6.0:
        recommendations.append("Major earthquake expected. Evacuate immediately and seek shelter in a safe area.")
        recommendations.append("Check emergency supplies and first aid kits.")
        recommendations.append("Listen to local news and follow official instructions.")
    elif magnitude >= 4.0:
        recommendations.append("Moderate earthquake expected. Secure heavy objects and prepare for potential aftershocks.")
        recommendations.append("Check structural integrity of your building.")
    else:
        recommendations.append("Minor earthquake expected. Be alert for any changes and ensure safety precautions are in place.")
    
    # Example recommendations based on location
    recommendations.append(f"Location: {latitude},{longitude}.Follow local emergency services' guidance.")
    
    return recommendations

def apply_recommendations(row):
    return generate_emergency_recommendations(row['Predicted'], row['latitude'],row['longitude'])

# Load your dataset
earthquake_data = pd.read_excel('magnitude_predictions.xlsx')

# Apply the recommendations
earthquake_data['Recommendation'] = earthquake_data.apply(apply_recommendations, axis=1)

# Save or display the results
print(earthquake_data.head())
earthquake_data.to_excel('earthquake_recommendations.xlsx', index=False)

print("Data with recommendations has been saved to earthquake_recommendations.xlsx")
# Z-score to detect anomalies
from scipy import stats

# Calculate Z-scores for each feature
z_scores = np.abs(stats.zscore(features))

# Set a threshold for anomaly detection
threshold = 3  # Usually, values greater than 3 are considered anomalies
earthquake_data['Anomaly_Z_Score'] = (z_scores > threshold).any(axis=1)

# Flag rows as anomaly or not based on Z-scores
anomalies_z = earthquake_data[earthquake_data['Anomaly_Z_Score'] == True]
print(f"Number of anomalies detected using Z-score: {len(anomalies_z)}")
from sklearn.ensemble import IsolationForest

# Isolation Forest model
features = features.loc[earthquake_data.index]

iso_forest = IsolationForest(contamination=0.05)
iso_forest_predictions = iso_forest.fit_predict(features)
earthquake_data['Anomaly_IsolationForest'] = iso_forest_predictions
  
# IsolationForest labels anomalies as -1 and normal points as 1
anomalies_iforest = earthquake_data[earthquake_data['Anomaly_IsolationForest'] == -1]
print(f"Number of anomalies detected using Isolation Forest: {len(anomalies_iforest)}")
from sklearn.neighbors import LocalOutlierFactor

# Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
earthquake_data['Anomaly_LOF'] = lof.fit_predict(features)

# LOF labels anomalies as -1 and normal points as 1
anomalies_lof = earthquake_data[earthquake_data['Anomaly_LOF'] == -1]
print(f"Number of anomalies detected using LOF: {len(anomalies_lof)}")
# Combine results from all methods
earthquake_data['Anomaly_Combined'] = (earthquake_data['Anomaly_Z_Score'] | 
                                       (earthquake_data['Anomaly_IsolationForest'] == -1) |
                                       (earthquake_data['Anomaly_LOF'] == -1))

# Extract the combined anomalies
combined_anomalies = earthquake_data[earthquake_data['Anomaly_Combined'] == True]

# Save anomalies to a new Excel file
combined_anomalies.to_excel('earthquake_anomalies.xlsx', index=False)
print(f"Anomalies saved to 'earthquake_anomalies.xlsx'.")

selected_features.set_index('time', inplace=True)

# Plot the time series of earthquake magnitudes
plt.figure(figsize=(10, 6))
sns.lineplot(x=selected_features.index, y=selected_features['mag'], marker='o', label='Magnitude')

# Highlight anomalies in red


np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt

# Load data
import pandas as pd
import matplotlib.pyplot as plt

# Load data
mag = pd.read_excel('magnitude_predictions.xlsx')
anam = pd.read_excel('earthquake_anomalies.xlsx')

# Extract columns
depth = mag['depth']
magnitudes = mag['Predicted']
anomalies = anam['Predicted']

# Create the main DataFrame
earthquake_data = pd.DataFrame({
    'depth': depth,
    'magnitude': magnitudes
})

# Reset index to use 'depth' for plotting
earthquake_data.set_index('depth', inplace=True)

# Ensure anomalies data is aligned with the main DataFrame
# Create an anomaly column with default values (0)
earthquake_data['Anomaly'] = 0

# Update anomalies based on index alignment
# Assuming `anam` contains index information and has depth column
# Align indices of anomalies with earthquake_data index
anomaly_indices = anam['depth']  # Use 'depth' if it exists or the appropriate column
earthquake_data.loc[anomaly_indices, 'Anomaly'] = 1  # Mark anomalies

# Plotting
plt.figure(figsize=(14, 7))

# Plot all magnitudes
plt.plot(earthquake_data.index, earthquake_data['magnitude'], label='Magnitude', alpha=0.7, color='blue')

# Highlight anomalies
anomalies_df = earthquake_data[earthquake_data['Anomaly'] == 1]
plt.scatter(anomalies_df.index, anomalies_df['magnitude'], color='red', label='Anomalies', zorder=5)

plt.xlabel('Depth')
plt.ylabel('Magnitude')
plt.title('Earthquake Magnitudes with Anomalies Highlighted')
plt.legend()
plt.grid(True)

# Save plot to a file
plt.savefig('earthquake_magnitudes_with_anomalies.png')

# Show plot
plt.show()
historical_data = st.file_uploader("Upload Historical Earthquake Data", type=["xlsx"])
if historical_data:
    historical_df = pd.read_excel(historical_data)
    st.write("Historical Data")
    st.dataframe(historical_df)

    # Plot comparison
    fig, ax = plt.subplots()
    ax.plot(earthquake_data['depth'], earthquake_data['magnitude'], label='Recent Data', alpha=0.7, color='blue')
    ax.plot(historical_df['depth'], historical_df['magnitude'], label='Historical Data', alpha=0.7, color='green')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Magnitude')
    ax.set_title('Comparison of Recent and Historical Earthquake Data')
    ax.legend()
    st.pyplot(fig)
 ########----------###########---------############----THANK YOU----------###########################-------------##########