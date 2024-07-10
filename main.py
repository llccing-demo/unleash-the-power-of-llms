import http.client
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")

url = "https://oneapi.gptnb.me/v1/completions"
conn = http.client.HTTPSConnection("oneapi.gptnb.me")

payload = json.dumps({
    "model": "gpt-3.5-turbo",
    #"model": "gpt-4-gizmo-g-IcWrQy2I9",
    "messages": [
        {"role": "system", "content": "you are a helpful assistant."},
        {"role": "user", "content": "who are you"},
    ]
})

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

print(payload)
print(headers)

conn.request('POST', '/v1/chat/completions', payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode('utf-8'))


#if __name__ == "__main__":
    #user_prompt = input("Enter your prompt: ")
    #result = call_openai_api(user_prompt)
    #print("Response from OpenAI:")
    #print(result)
