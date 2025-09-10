import joblib

def load_models():
    """Load all trained models + scaler"""
    rf = joblib.load("./models/randomforest.pkl")
    xgb = joblib.load("./models/xgboost.pkl")
    svm = joblib.load("./models/svm.pkl")
    voting = joblib.load("./models/voting_model.pkl")
    scaler = joblib.load("./models/scaler.pkl")
    return rf, xgb, svm, voting, scaler
