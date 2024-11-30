import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv('data/house.csv')

# Preprocess the dataset
X = data.drop('price', axis = 1)
y = data['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load the saved model
model = joblib.load('model/house.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test) 
print(f"accuracy_score : {accuracy:.2f}")