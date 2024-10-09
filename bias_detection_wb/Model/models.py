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
            # Format the prompt with the question
            prompt = self.prompt_template.format(question=question)
            
            # Create the chat completion using the OpenAI API
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs JSON responses. Do not include any explanations or additional text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            # Get the assistant's reply
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

                # Parse the JSON content
                try:
                    parsed = json.loads(json_content)
                except json.JSONDecodeError as e:
                    print(f"JSON decoding failed: {e}")
                    print("JSON content was:")
                    print(json_content)
                    return None

                print("Parsed JSON before cleaning:")
                print(parsed)

                # Clean up the keys (remove extra quotes and whitespace)
                parsed = {key.strip().strip('"'): value for key, value in parsed.items()}
                print("Parsed JSON after cleaning:")
                print(parsed)

                # Map the model's output keys to the desired output keys
                output = {}

                # Map 'debias_question' to 'predicted_debias_question'
                output['predicted_debias_question'] = parsed.get('debias_question', None)
                # Map 'bias_score' to 'predicted_bias_score'
                output['predicted_bias_score'] = parsed.get('bias_score', None)
                # Map 'bias_axes' to 'predicted_bias_axes'
                output['predicted_bias_axes'] = parsed.get('bias_axes', None)
                # Map 'bias_explanation' to 'predicted_bias_explanation'
                output['predicted_bias_explanation'] = parsed.get('bias_explanation', None)

                # Add 'question' to the output
                output['question'] = question

                # Include 'debiased_text' for AgentMetrics scorer
                output['debiased_text'] = output['predicted_debias_question'] or ""

                # Ensure 'predicted_bias_score' is a number
                try:
                    output['predicted_bias_score'] = float(output['predicted_bias_score'])
                except (ValueError, TypeError):
                    print(f"Invalid predicted_bias_score value: {output['predicted_bias_score']}")
                    output['predicted_bias_score'] = 0.0  # Default to 0.0

                # Ensure 'predicted_bias_axes' is a list
                if not isinstance(output['predicted_bias_axes'], list):
                    print(f"predicted_bias_axes is not a list: {output['predicted_bias_axes']}")
                    output['predicted_bias_axes'] = []  # Default to empty list

                return output
            else:
                print("No JSON content found in model response.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred in predict method: {type(e).__name__}: {e}")
            return None
