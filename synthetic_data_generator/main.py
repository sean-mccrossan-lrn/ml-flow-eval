import os
import openai
import pandas as pd
import argparse
import random
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai.chat_models import ChatOpenAI

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")

# Define the schema for synthetic data
class DEIQuestions(BaseModel):
    bias_axes: Optional[str] = None
    question: str
    question_type: str
    bias_score: float
    bias_label: bool
    bias_explanation: Optional[str] = None
    debias_question: Optional[str] = None

# Function to load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Convert DataFrame rows to examples
def convert_to_examples(df):
    examples = []
    for _, row in df.iterrows():
        examples.append({
            "example": (
                f"Bias Axes: {row['bias_axes']}, Question: {row['question']}, "
                f"Question Type: {row['question_type']}, Bias Score: {row['bias_score']}, "
                f"Bias Label: {row['bias_label']}, Bias Explanation: {row['bias_explanation']}, "
                f"Debias Question: {row['debias_question']}"
            )
        })
    return examples

# Function to create the data generator and template
def create_synthetic_data_generator(examples):
    # Define the template for generating synthetic data
    OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")
    
    prompt_template = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["subject", "extra"],
        example_prompt=OPENAI_TEMPLATE,
    )

    # Create the synthetic data generator
    synthetic_data_generator = create_openai_data_generator(
        output_schema=DEIQuestions,
        llm=ChatOpenAI(temperature=1),
        prompt=prompt_template,
    )
    return synthetic_data_generator

# Function to generate synthetic data and save to CSV
def generate_synthetic_data(synthetic_data_generator, examples, output_csv):
    all_results = []
    for llm_run in range(1, 31):  # Loop 30 times
        # Take a random sample of 10 examples for few-shot prompting
        print(llm_run)
        sampled_examples = random.sample(examples, 10)
        synthetic_data_generator.llm_chain.prompt.examples = sampled_examples

        # Generate 10 synthetic examples
        synthetic_results = synthetic_data_generator.generate(
            subject="dei questions",
            extra=(
                "Generate diverse examples with unique bias explanations. "
                "Make sure questions span different contexts and biases. "
                "These should be questions aimed at school children from age 8-18"
            ),
            runs=10,
        )

        # Add llm_run and timestamp to each result
        timestamp = datetime.now().isoformat()
        for item in synthetic_results:
            item_dict = item.dict()
            item_dict['llm_run'] = llm_run
            item_dict['timestamp'] = timestamp
            all_results.append(item_dict)

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(all_results)

    # Save to CSV, appending if the file exists
    if not os.path.isfile(output_csv):
        results_df.to_csv(output_csv, index=False)
    else:
        results_df.to_csv(output_csv, mode='a', header=False, index=False)

    print(f"Data saved to {output_csv}")

# Main entry point for the script
if __name__ == "__main__":
    # Parse command-line arguments for the output CSV filename
    parser = argparse.ArgumentParser(description="Generate synthetic DEI questions.")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file name. If the file exists, data will be appended."
    )
    args = parser.parse_args()

    output_csv = args.output_csv

    # File path to your CSV
    input_csv = '/Users/seanmccrossan/Desktop/real_examples_dei_corrected.csv'

    # Load the CSV file into a DataFrame
    df = load_data(input_csv)

    # Convert the DataFrame rows to examples
    examples = convert_to_examples(df)

    # Ensure we have enough examples for sampling
    if len(examples) < 10:
        raise ValueError("Not enough examples to sample from. Please provide at least 10 examples.")

    # Create the synthetic data generator
    synthetic_data_generator = create_synthetic_data_generator(examples)

    # Generate synthetic data and save to CSV
    generate_synthetic_data(synthetic_data_generator, examples, output_csv)
