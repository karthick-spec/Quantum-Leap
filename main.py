from flask import Flask
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from datetime import datetime
from textblob import TextBlob  # For sentiment analysis

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('firebase.json')  # Replace with your Firebase service account key path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://quatum-leap-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Your database URL
})

# Helper function to convert int64 to int
def convert_int64_to_int(data):
    if isinstance(data, dict):
        return {key: convert_int64_to_int(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_int64_to_int(item) for item in data]
    elif isinstance(data, np.int64):
        return int(data)
    return data

# Load the saved model, scaler, and feature columns
with open('model/prediction_model.pkl', 'rb') as f:
    rf_classifier = joblib.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

with open('model/feature_columns.pkl', 'rb') as f:
    feature_columns = joblib.load(f)

# Map Firebase data to feature columns
def map_to_feature_columns(firebase_data):
    feature_column_mapping = {
        'Age': 'Age',
        'Gender': 'Gender',
        'Married': 'Married',
        'Homeownership': 'Homeownership',
        'Number of Dependents': 'Number_of_Dependents',
        'Owns Computer': 'Owns_Computer',
        'Tenure in Months': 'Tenure_in_Months',
        'Unanswered Calls': 'Unanswered_Calls',
        'Off Peak Calls In/Out': 'Off_Peak_Calls_In_Out',
        'Overage Minutes': 'Overage_Minutes',
        'Monthly Revenue': 'Monthly_Revenue',
        'Customer Care Calls': 'Customer_Care_Calls',
        'Monthly Charges': 'Monthly_Charges',
        'Monthly Minutes': 'Monthly_Minutes',
        'Online Security': 'Online_Security',
        'Number of Referrals': 'Number_of_Referrals',
        'Payment Method': 'Payment_Method',
        'Roaming Calls': 'Roaming_Calls',
        'Has Credit Card': 'Has_Credit_Card',
        'Retention Calls': 'Retention_Calls',
        'Blocked Calls': 'Blocked_Calls',
        'Overage Fee': 'Overage_Fee',
        'Call Forwarding Calls': 'Call_Forwarding_Calls',
        'Income Group': 'Income_Group',
        'Contract Type': 'Contract_Type',
        'Internet Service': 'Internet_Service',
        'Device Protection Plan': 'Device_Protection_Plan',
        'Online Backup': 'Online_Backup',
        'Streaming TV': 'Streaming_TV',
        'Streaming Movies': 'Streaming_Movies',
        'Premium Tech Support': 'Premium_Tech_Support',
        'Phone Service': 'Phone_Service',
        'Owns Motorcycle': 'Owns_Motorcycle',
        'Retention Offers Accepted': 'Retention_Offers_Accepted',
        'Unlimited Data': 'Unlimited_Data',
        'Account Length': 'Account_Length',
        'Total Charges': 'Total_Charges',
        'Streaming Music': 'Streaming_Music',
        'Total Refunds': 'Total_Refunds',
        'Avg Monthly Long Distance Charges': 'Avg_Monthly_Long_Distance_Charges',
        'Data Usage': 'Data_Usage',
        'Call Waiting Calls': 'Call_Waiting_Calls',
        'Multiple Lines': 'Multiple_Lines',
        'Dropped Calls': 'Dropped_Calls',
        'Paperless Billing': 'Paperless_Billing',
        'Peak Calls In/Out': 'Peak_Calls_In_Out',
        'Director Assisted Calls': 'Director_Assisted_Calls',
        'Avg Monthly GB Download': 'Avg_Monthly_GB_Download',
        'Credit Rating': 'Credit_Rating'
    }

    mapped_data = {}
    for feature, column in feature_column_mapping.items():
        mapped_data[feature] = firebase_data.get(column, 0)  # Default to 0 for missing fields
    return mapped_data

# Simulate progress tracking and update it in Firebase
def simulate_progress(firebase_ref):
    for progress in range(0, 101, 10):
        firebase_ref.set({'progress': progress})
        time.sleep(0.5)

# Perform prediction and save the result in Firebase
def perform_prediction(data):
    try:
        if not isinstance(data, dict):
            raise ValueError("Expected a dictionary but got a different data type")

        ref_progress = db.reference('/prediction_progress')
        simulate_progress(ref_progress)

        mapped_data = map_to_feature_columns(data)
        sample_df = pd.DataFrame([mapped_data])
        sample_df = sample_df[feature_columns]
        sample_scaled = scaler.transform(sample_df)

        prediction = rf_classifier.predict(sample_scaled)[0]
        timestamp = datetime.now().isoformat()

        ref_prediction = db.reference('/predictions')
        ref_prediction.push({'prediction': int(prediction), 'timestamp': timestamp})
    except Exception as e:
        print(f"Error in prediction: {e}")

# Analyze sentiment and store it in Firebase
def analyze_and_store_feedback(feedback_data):
    try:
        for feedback_id, feedback_entry in feedback_data.items():
            text = feedback_entry.get('text', '')
            if text:
                sentiment_score = get_sentiment_score(text)
                print(f"Feedback: {text}, Sentiment Score: {sentiment_score}")
                
                # Store the sentiment score in Firebase
                ref_feedback_sentiment = db.reference('/feedback_analysis')
                ref_feedback_sentiment.child(feedback_id).set({
                    'sentiment_score': sentiment_score,
                    'original_feedback': text,
                    'timestamp': datetime.now().isoformat()
                })
    except Exception as e:
        print(f"Error in feedback analysis: {e}")

# Helper function to perform sentiment analysis
def get_sentiment_score(text):
    try:
        blob = TextBlob(text)
        # Polarity ranges from -1 (negative) to 1 (positive)
        return blob.sentiment.polarity
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0  # Default score for errors

# Firebase listener for customer data and feedback
def listener(event):
    try:
        # Listen to customer data
        ref_data = db.reference('/customer_data')
        full_data = convert_int64_to_int(ref_data.get())

        if isinstance(full_data, dict):
            perform_prediction(full_data)

        # Listen to feedback data
        ref_feedback = db.reference('/customer_feedback')
        feedback_data = ref_feedback.get()

        if feedback_data:
            analyze_and_store_feedback(feedback_data)
        else:
            print("No feedback data available.")
    except Exception as e:
        print(f"Listener error: {e}")

# Watch for changes in Firebase
def watch_for_changes():
    ref_data = db.reference('/customer_data')
    ref_data.listen(listener)

    ref_feedback = db.reference('/customer_feedback')
    ref_feedback.listen(listener)

# Start the app
if __name__ == '__main__':
    watch_for_changes()
    app.run(debug=True, host='0.0.0.0', port=5000)
