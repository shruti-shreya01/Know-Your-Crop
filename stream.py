import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq

load_dotenv()
GROQ_API_KEY=os.getenv('GROQ_API_KEY')

# Defining LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# Load the trained CNN model
model = load_model('CNN_Plant_Recognition.h5')

# Dictionary to map predictions to crop names (update it based on your dataset)
crop_labels = {
    0: 'jute',  # Replace with actual crop names
    1: 'sugarcane',
    2: 'wheat',
    3: 'rice',
    4: 'maize',
}

# Set up the Streamlit app
st.title("Know your Crop")

# Instructions for the user
st.write("Upload an image of a crop leaf for disease prediction.")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)

    # Display the image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for prediction
    img = img.resize((224, 224))  # Resize to match the input size of the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction using the model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability

    # Display the prediction
    st.write(f"Prediction: {crop_labels[predicted_class]}")
    prompt = crop_labels[predicted_class]
    # messages = [
    #     ("system", "You are a helpful assistant who give the information about the crop and the suggest about crop production and yield based on the given crop."),
    #     ("human", prompt),
    # ]
    messages = [
    (
        "system",
        "You are a helpful assistant that generates recommendations for crop care, pest control, and fertilizer based on the predicted crop type. Your role is to: \n"
        "1. Provide season-specific care tips for the given crop.\n"
        "2. Suggest appropriate fertilizers or pesticides to maximize yield.\n"
        "3. Offer irrigation and planting advice tailored to regional conditions and crop requirements.\n"
        "Use clear and actionable language to guide the user effectively."
    ),
    ("human", prompt),
]

    
    # Assuming 'llm.invoke' generates the response from the model
    ai_msg = llm.invoke(messages)
    
    if st.button("Know more"):
        if prompt:
            with st.spinner("Result is generating, please wait..."):
                ai_msg = llm.invoke(messages)
                st.write(f"LLM Response: {ai_msg.content}")