import streamlit as st
import pickle
import numpy as np
import requests

st.title('Welcome OVA Breast Cancer Predictor App')

# Load your trained model
model_path = "logistic_regression_model.pkl"  
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Chatbot API function
CHATBOT_API_URL = "'https://trypromptly.com/api/apps/d353d815-2767-4709-b64f-ba919ada371a/run'"

def get_chatbot_response(user_input):
    payload = {'message': user_input}
    response = requests.post(CHATBOT_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get('response')
    else:
        return "Sorry, I couldn't process your request right now."

# Function to make predictions
def predict_cancer(input_data):
    # Transform the input into the correct shape for the model
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return "Malignant" if prediction[0] == 1 else "Benign"

# Streamlit interface
def main():
    st.title("Breast Cancer Prediction App")
    
    # User inputs for model prediction
    st.header("Input Features")
    mean_radius = st.number_input("Mean Radius", min_value=0.0)
    mean_texture = st.number_input("Mean Texture", min_value=0.0)
    mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0)
    mean_area = st.number_input("Mean Area", min_value=0.0)
    mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0)
    
    # Add more input fields as needed based on your model's features
    # For example: st.number_input for standard error, worst values, etc.
    
    if st.button("Predict"):
        # Collect input data for prediction
        input_data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
        result = predict_cancer(input_data)
        st.success(f"The prediction is: {result}")
    
    # Chatbot section
    st.header("Ask the Chatbot")
    user_message = st.text_input("Enter your question for the chatbot:")
    
    if st.button("Get Chatbot Response"):
        chatbot_response = get_chatbot_response(user_message)
        st.write(f"Chatbot: {chatbot_response}")

if __name__ == '__main__':
    main()
