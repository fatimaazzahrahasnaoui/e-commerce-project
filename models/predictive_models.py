import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Load and preprocess data
def load_and_preprocess_data():
    # Use the absolute path to your CSV file
    data_path = r"C:\Users\Hp 14s\Desktop\e-commerce - Copie\e-commerce\data.csv\DATA\data_commerce.csv"
    data = pd.read_csv(data_path)
    
    # Drop rows with missing values (optional)
    data = data.dropna()
    
    # Encode categorical variables
    le_country = LabelEncoder()
    data['Country'] = le_country.fit_transform(data['Country'])
    
    le_category = LabelEncoder()
    data['Category'] = le_category.fit_transform(data['Category'])
    
    # Define features and target
    features = data[['Quantity', 'UnitPrice', 'Age', 'Stock', 'Rating', 'Country']]
    target = data['Status']  # Assuming Status is the target column
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le_country

# Train and save the model
def train_and_save_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Ensure the models directory exists
    models_directory = r"C:\Users\Hp 14s\Desktop\e-commerce - Copie\e-commerce\data.csv\models"
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    
    # Save the model to a .pkl file
    model_path = os.path.join(models_directory, "logistic_regression_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved as {model_path}")
    return model

# Main execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le_country = load_and_preprocess_data()
    model = train_and_save_model(X_train, y_train)
