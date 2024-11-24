import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = ""

client = OpenAI()

def queryGPT(prompt):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Given the following prompt, tell me if the intent of that prompt is 'recommend', 'compare', or 'conversational'. "
                       "Only respond with 'recommend' or 'compare' if the prompt is about vehicles. "
                       f"DO NOT add anything, just respond with one of the three words: {prompt}"
        }
    ]
)
    # Accessing the response as an object attribute
    response = completion.choices[0].message.content
    return response

def queryGPTCustom(prompt, instructions):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"{instructions}: {prompt}"
            }
        ]
    )

    # Accessing the response as an object attribute
    response = completion.choices[0].message.content

    return response