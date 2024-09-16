# models.py

import openai
import json
import weave
import re

class BiasModel(weave.Model):
    model_name: str
    prompt_template: str

    @weave.op()
    def predict(self, Question: str) -> dict:
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs JSON responses. Do not include any explanations or additional text."
                    },
                    {"role": "user", "content": self.prompt_template.format(sentence=Question)}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError("No response from model")

            # print(f"Model response content:\n{content}")  # Debug print

            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                parsed = json.loads(json_content)
                return parsed
            else:
                print("No JSON content found in model response.")
                return None
        except Exception as e:
            print(f"Error in predict method: {e}")
            return None
