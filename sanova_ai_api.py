from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ✅ Step 1: Sample Medical Symptom Dataset
expanded_symptom_data = [
    ("Fever, cough, fatigue", "Flu"),
    ("Fever, dry cough, loss of taste, shortness of breath", "COVID-19"),
    ("Chest pain, shortness of breath, dizziness, sweating", "Heart Attack"),
    ("Frequent urination, increased thirst, blurred vision, weight loss", "Diabetes"),
    ("Joint pain, swelling, stiffness, morning stiffness", "Arthritis"),
    ("Runny nose, sneezing, itchy eyes, congestion", "Allergy"),
    ("Headache, nausea, light sensitivity, aura", "Migraine"),
    ("Sore throat, swollen lymph nodes, white patches on tonsils", "Strep Throat"),
    ("Abdominal pain, nausea, vomiting, loss of appetite", "Appendicitis"),
    ("Fatigue, pale skin, shortness of breath, dizziness", "Anemia"),
]

df_symptoms = pd.DataFrame(expanded_symptom_data, columns=["symptoms", "condition"])

# ✅ Step 2: Train the Symptom Checker Model
symptom_vectorizer = TfidfVectorizer()
X = symptom_vectorizer.fit_transform(df_symptoms["symptoms"])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_symptoms["condition"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train, y_train)

# ✅ Step 3: Define Function for Predicting Condition
def predict_condition_api(symptom_input):
    input_vector = symptom_vectorizer.transform([symptom_input])
    predicted_label = model_logistic.predict(input_vector)[0]
    return label_encoder.inverse_transform([predicted_label])[0]

# ✅ Step 4: Create API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms")
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    
    predicted_condition = predict_condition_api(symptoms)
    return jsonify({"condition": predicted_condition})

# ✅ Step 5: Run API
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Get PORT from Railway
    app.run(host="0.0.0.0", port=port, debug=True)

