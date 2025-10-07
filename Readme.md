📘 Project Description

PetImage-Detection-GenerativeAI is a FastAPI-based AI-powered application that analyzes pet images and health data using Google Gemini’s Generative AI.
The system identifies the breed of a pet from an uploaded image (or accepts a manually provided breed name), compares the pet’s physical and lifestyle parameters (like weight, height, grooming, and walking frequency) with normal healthy ranges, and then provides personalized health insights and care suggestions.

🚀 Features

🧠 AI-based Pet Breed Detection — Automatically identifies breed from uploaded images using Google Gemini AI.

📊 Health Metrics Evaluation — Compares user-provided data with breed-specific healthy ranges.

💬 Smart Suggestions — Provides AI-generated personalized care tips based on detected abnormalities.

🐶 Multi-modal Input — Accepts both image and text-based pet data.

⚙️ FastAPI Integration — Fully asynchronous backend for fast and scalable API responses.

🌍 Environment Variable Support — Secure API key management with .env integration.

🧩 Tech Stack

Python 3.10+

FastAPI (Backend Framework)

Google Generative AI (Gemini API)

Pydantic (Data validation)

dotenv (Environment management)

Run this project in your machine

step 1 --> create venv Environment in your machine using:

python -m venv venv

step 2 --> activate the virtual Environment:

.\venv\Scripts\activate

step 3 --> now install all the libraries and required packages:

pip install fastapi uvicorn python-multipart google-generativeai python-dotenv

step 4 --> now add your api from Gemini api key from ai studio

example: GEMINI_API_KEY= AIza............................X20

step 5 --> now start app on local host:

uvicorn main:app --reload

after running this command you will see Uvicorn running on http://127.0.0.1:8000
add /docs in this local host link--> http://127.0.0.1:8000/docs

now you can try this on your own device