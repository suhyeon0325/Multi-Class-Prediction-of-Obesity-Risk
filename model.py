import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump, load
import os

DATA_PATH = './data/train.csv'
data = pd.read_csv(DATA_PATH)

# Separate features and target variable
X = data.drop(['id', 'NObeyesdad'], axis=1)
y = data['NObeyesdad']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
num_columns = X.select_dtypes(include=['float64']).columns
cat_columns = X.select_dtypes(include=['object']).columns

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns)
    ]
)

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# 모델 저장
model_directory = 'model'

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# RESTful API 방식으로 모델을 내보내기
model_path = os.path.join(model_directory, 'NObeyesdad_prediction_pipeline.joblib')
dump(pipeline, model_path)