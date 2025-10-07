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