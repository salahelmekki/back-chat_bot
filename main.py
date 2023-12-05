from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import openai

app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


# Model for request payload
class RequestPayload(BaseModel):
    messages: list


# Route to handle the OpenAI API request
@app.post("/openai")
async def openai_request(request_payload: RequestPayload):
    user_message = request_payload.messages[0]['content']

    # Extract keywords using NLTK
    keywords = extract_keywords(user_message)

    # Call OpenAI API to get a response
    openai_response = call_openai_api(user_message)

    # Combine NLTK and OpenAI responses
    combined_response = f"NLTK Keywords: {keywords}\n\nOpenAI Response: {openai_response}"

    return {"assistant_reply": combined_response}


# NLTK function to extract keywords
def extract_keywords(question):
    words = word_tokenize(question)
    keywords = [porter.stem(word) for word in words if word.lower() not in stop_words]
    return keywords


# Dummy function to simulate OpenAI API call
def call_openai_api(question):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # Adjust the model as needed
        prompt=question,
        temperature=0.7,
        max_tokens=150,
        stop=None
    )
    return response.choices[0].text.strip()
