import pandas as pd
import joblib
import pytest
import os
import sys

# Define path
MODEL_PATH = 'churn_pipeline.pkl'
FEATURES_PATH = 'feature_types.pkl'
TEST_DATA_PATH = 'test_sample.csv'

def test_model_files_exist():
    """Ensure that the primary pipeline artifacts exist and are loaded."""
    assert os.path.exists(MODEL_PATH), "Trained pipeline file is missing."
    assert os.path.exists(FEATURES_PATH), "Feature types file is missing."
    assert os.path.exists(TEST_DATA_PATH), "Test sample dataset is missing."

def test_pipeline_prediction():
    """Ensure the pipeline outputs expected binary values from the test sample."""
    pipeline = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # We must drop the target because the pipeline shouldn't see it
    if 'Churn' in test_df.columns:
        X_test = test_df.drop(columns=['Churn'])
    else:
        X_test = test_df
        
    # The pipeline should handle the dataframe correctly
    try:
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
    except Exception as e:
        pytest.fail(f"Pipeline prediction failed with error: {str(e)}")
        
    assert len(predictions) == len(X_test), "Mismatch in prediction output length."
    assert len(probabilities) == len(X_test), "Mismatch in prediction probability output length."
    assert all(pred in [0, 1] for pred in predictions), "Predictions must be binary (0 or 1)."
    
def test_streamlit_app_compilation():
    """Simple smoke test to ensure no syntax/import errors in app.py"""
    try:
        import app
        assert app is not None
    except Exception as e:
        pytest.fail(f"Streamlit app failed to load or compile: {str(e)}")

# Add a procedural run block so we don't *have* to use pytest purely from cli.
if __name__ == "__main__":
    print("Running Tests Manually...")
    test_model_files_exist()
    print("Files exist.")
    test_pipeline_prediction()
    print("Pipeline predicts cleanly on test sample.")
    test_streamlit_app_compilation()
    print("App loads cleanly.")
    print("\n✅ All logic tests passed successfully!")
