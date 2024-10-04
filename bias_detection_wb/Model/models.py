# models.py

import openai
import json
import weave
import re

class BiasModel(weave.Model):
    model_name: str
    prompt_template: str

    @weave.op()
    def predict(self, question: str) -> dict:
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs JSON responses. Do not include any explanations or additional text."
                    },
                    {"role": "user", "content": self.prompt_template.format(question=question)}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError("No response from model")

            print("Model response content:")
            print(content)

            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                print("Extracted JSON content:")
                print(json_content)

                parsed = json.loads(json_content)
                print("Parsed JSON before cleaning:")
                print(parsed)

                # Clean up the keys
                parsed = {key.strip().strip('"'): value for key, value in parsed.items()}
                print("Parsed JSON after cleaning:")
                print(parsed)

                # Ensure required fields are present
                required_fields = ['debiased_text', 'bias_score', 'bias_axes']
                for field in required_fields:
                    if field not in parsed:
                        print(f"Missing field '{field}' in model response.")
                        return None

                # Add 'question' to the output
                parsed['question'] = question

                return parsed
            else:
                print("No JSON content found in model response.")
                return None
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            print("JSON content was:")
            print(json_content)
            return None
        except KeyError as e:
            print(f"KeyError: Missing key {e} in parsed JSON.")
            print("Parsed JSON keys:")
            print(parsed.keys())
            return None
        except Exception as e:
            print(f"An unexpected error occurred in predict method: {type(e).__name__}: {e}")
            return None
