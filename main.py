# main.py (FastAPI backend)
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List

app = FastAPI()

# CORS middleware for allowing requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust accordingly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API key (replace 'YOUR_OPENAI_API_KEY' with your actual API key)
client = OpenAI(api_key='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')

class UserMessage(BaseModel):
    content: str
    role: str

@app.post("/openai")
async def chat_endpoint(messages: List[UserMessage]):
    try:
        # Prepare messages for the Chat API
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Make a request to OpenAI Chat API
        response = client.Chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={"type": "json_object"},
            messages=chat_messages
        )

        # Extract the assistant's reply from the response
        assistant_reply = response['choices'][0]['message']['content']

        return {"assistant_reply": assistant_reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")


