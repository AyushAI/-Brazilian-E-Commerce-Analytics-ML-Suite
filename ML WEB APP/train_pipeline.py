import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading dataset...")
    df = pd.read_csv('final_outlierTreated.csv')
    
    # We drop columns that are too granular or geographically specific for our simple app
    # customer_unique_id: unique identifier
    # customer_city: 4000+ unique values, hard for UI select box
    # customer_state: could be kept, but let's keep it simple
    # geolocation_lat, geolocation_lng: geographical coordinates
    drop_cols = ['customer_unique_id', 'customer_city', 'customer_state', 'geolocation_lat', 'geolocation_lng']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Target column
    target = 'Churn'
    
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Training XGBoost Pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating Model...")
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the pipeline
    print("Saving the pipeline to churn_pipeline.pkl...")
    joblib.dump(pipeline, 'churn_pipeline.pkl')
    
    # Save feature metadata for Streamlit App
    print("Saving feature metadata...")
    joblib.dump({'numerical': numeric_features, 'categorical': categorical_features}, 'feature_types.pkl')
    
    # Save unique categories for the selectboxes
    category_values = {}
    for cat_col in categorical_features:
        category_values[cat_col] = sorted(df[cat_col].dropna().unique().tolist())
        
    joblib.dump(category_values, 'category_values.pkl')
    
    print("Training complete! Model and metadata saved successfully.")

if __name__ == "__main__":
    main()
