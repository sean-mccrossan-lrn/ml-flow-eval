import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
print(OPEN_AI_KEY)
client = OpenAI(api_key=OPEN_AI_KEY)

def call_openai_model(model_config, prompt, data):
    """
    Function to call the OpenAI model for translation using GPT-4.
    """
    try:
        system_prompt = prompt
        combined_text = f"Here's a text to translate: {data}"

        response = client.chat.completions.create(
            model=model_config.actual_model_name,  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_text}
            ],
            temperature=model_config.temperature,
            max_tokens=4000  
        )
        return response
    except Exception as e:
        print(f"Error with OpenAI model: {e}")
        return None
