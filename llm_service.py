import requests
import json
from config import FIREWORKS_API_KEY, FIREWORKS_URL, MODEL_NAME, DEFAULT_PARAMS

def call_deepseek(user_prompt):
    # CORRECT
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        **DEFAULT_PARAMS,
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": user_prompt}]
    }
    response = requests.post(FIREWORKS_URL, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']