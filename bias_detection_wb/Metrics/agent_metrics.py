# agent_metrics.py

import weave
from weave.flow.scorer import Scorer
from typing import Optional, Any, List, Dict
import openai
import json
from pydantic import BaseModel
import numpy as np  # Import numpy to check for NaN

class AgentMetrics(Scorer, BaseModel):
    model_name: str
    openai_api_key: str
    dataset_mapping: Dict[str, str]  # Mapping from question to debias_question

    @classmethod
    def from_dataset(cls, model_name: str, openai_api_key: str, dataset: List[Dict[str, Any]]):
        """
        Initializes the AgentMetrics scorer with a mapping from question to debias_question.

        Args:
            model_name (str): The name of the OpenAI model to use for evaluation.
            openai_api_key (str): Your OpenAI API key.
            dataset (List[Dict[str, Any]]): The evaluation dataset.

        Returns:
            AgentMetrics: An instance of AgentMetrics with the dataset mapping.
        """
        mapping = {}
        for example in dataset:
            question = example.get('question')
            debias_question = example.get('debias_question')

            # Skip entries where question is None or NaN
            if question is None or (isinstance(question, float) and np.isnan(question)):
                print(f"Skipping entry due to invalid question.")
                continue

            # Check if debias_question is a valid string
            if isinstance(debias_question, str):
                mapping[question] = debias_question
            elif debias_question is not None and not (isinstance(debias_question, float) and np.isnan(debias_question)):
                mapping[question] = str(debias_question)
            else:
                print(f"Skipping entry with question: '{question}' due to invalid debias_question.")
                # If debias_question is invalid, skip this entry
                continue

        return cls(model_name=model_name, openai_api_key=openai_api_key, dataset_mapping=mapping)

    @weave.op()
    async def score(self, model_output: dict) -> dict:
        """
        Scores the model output based on various metrics.
        """
        if model_output is None:
            print("Model output is None. Skipping evaluation.")
            return {
                "answer_relevancy": None,
                "correctness": None,
                "hallucination": None,
                "contextual_relevancy": None
            }

        # Proceed with extracting fields
        question = model_output.get('question')
        debiased_text = model_output.get('debiased_text')
        debias_question = self.dataset_mapping.get(question)

        if not all([question, debias_question, debiased_text]):
            print(f"Missing required fields for question: '{question}'. Skipping evaluation.")
            return {
                "answer_relevancy": None,
                "correctness": None,
                "hallucination": None,
                "contextual_relevancy": None
            }

        # Initialize OpenAI API
        openai.api_key = self.openai_api_key

        # Prepare the prompt for evaluation
        evaluation_prompt = f"""
You are an AI assistant tasked with evaluating the quality of debiased questions generated by an LLM. Given the original question, the debiased question generated by the model, and the reference debiased question, evaluate the following metrics:

1. **Answer Relevancy**: Is the debiased question relevant to the original question?
2. **Correctness**: Does the debiased question correctly remove bias from the original question while preserving its intent?
3. **Hallucination**: Does the debiased question introduce any information not present in the original question?
4. **Contextual Relevancy**: Is the debiased question contextually appropriate and coherent?

Provide your evaluation as a JSON object with boolean values for each metric, like this:

{{
  "answer_relevancy": true/false,
  "correctness": true/false,
  "hallucination": true/false,
  "contextual_relevancy": true/false
}}

Original Question: "{question}"
Debiased Question Generated by Model: "{debiased_text}"
Reference Debiased Question: "{debias_question}"

Your evaluation:
"""

        try:
            # Call the OpenAI API to get the evaluation
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON from the response
            json_content = self.extract_json(content)
            if json_content:
                evaluation = json.loads(json_content)
                return evaluation
            else:
                print("No JSON content found in evaluation response.")
                return {
                    "answer_relevancy": None,
                    "correctness": None,
                    "hallucination": None,
                    "contextual_relevancy": None
                }
        except Exception as e:
            print(f"Error in AgentMetrics score method: {e}")
            return {
                "answer_relevancy": None,
                "correctness": None,
                "hallucination": None,
                "contextual_relevancy": None
            }

    def extract_json(self, text: str) -> Optional[str]:
        """Utility function to extract JSON from text."""
        try:
            json_start = text.index('{')
            json_end = text.rindex('}') + 1
            json_str = text[json_start:json_end]
            return json_str
        except ValueError:
            return None

    @weave.op()
    def summarize(self, score_rows: List[Any]) -> Optional[dict]:
        """Aggregate all the scores that are calculated for each row."""
        metrics = ["answer_relevancy", "correctness", "hallucination", "contextual_relevancy"]
        summary = {}
        for metric in metrics:
            valid_data = [x.get(metric) for x in score_rows if x.get(metric) is not None]
            count_true = valid_data.count(True)
            int_data = [int(bool(x)) for x in valid_data]
            sample_mean = sum(int_data) / len(int_data) if int_data else 0
            sample_variance = sum((x - sample_mean) ** 2 for x in int_data) / len(int_data) if int_data else 0
            sample_error = (sample_variance / len(int_data)) ** 0.5 if int_data else 0
            summary[metric] = {
                "true_count": count_true,
                "true_fraction": sample_mean,
                "stderr": sample_error,
            }
        return summary
