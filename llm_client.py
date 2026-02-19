"""
llm_client.py
Handles LLM API interaction via OpenRouter using httpx.
"""

import os
import httpx

api_key = os.getenv("OPENROUTER_API_KEY")


async def generate_response(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    data = {
        "model": "openai/gpt-3.5-turbo",  # or openai/gpt-4o-mini
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Scratch-RAG-System"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers)

    response.raise_for_status()  # raise error if bad response
    result = response.json()

    return result["choices"][0]["message"]["content"]
