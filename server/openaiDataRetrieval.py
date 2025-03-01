
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def get_fungus_info_from_chatgpt(species):
    # Instruct the model to output a JSON response with the required fields.
    prompt = (
        f"Provide detailed information about the fungus '{species}' in the following JSON format without any additional commentary:\n\n"
        "{\n"
        '  "summary": "A brief summary of the fungus.",\n'
        '  "habitat": "Location and habitat information.",\n'
        '  "edibility": "Edibility information.",\n'
        '  "genus": "Genus of the fungus.",\n'
        '  "characteristics": ["List", "of", "cool", "characteristics"]\n'
        "}"
    )
    
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or another model you have access to
            store=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable mycology expert. "
                        "Return all information as valid JSON using the provided format."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=250,
            n=1,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        response_text = completion.choices[0].message.content
        # Parse the JSON response.
        data = json.loads(response_text)
        return data
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return {
            "summary": "Detailed information currently unavailable.",
            "habitat": "",
            "edibility": "",
            "genus": "",
            "characteristics": []
        }
