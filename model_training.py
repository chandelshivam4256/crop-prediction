# model_training.py

# 1️⃣ Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 2️⃣ Load dataset
# Replace 'crop_dataset.csv' with your actual CSV filename
df = pd.read_csv('crop_dataset.csv')

# 3️⃣ Prepare features and target
# Replace column names with your dataset columns
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']  # The column you want to predict

# 4️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6️⃣ Evaluate model
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")

# 7️⃣ Save the trained model
pickle.dump(model, open('crop_model.pkl', 'wb'))
print("✅ Model saved as 'crop_model.pkl'")
