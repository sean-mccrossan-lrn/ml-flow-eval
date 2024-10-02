import os
import openai
import pandas as pd
import argparse
import random
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ValidationError
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain.chat_models import ChatOpenAI

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
        llm=ChatOpenAI(temperature=0.8),
        prompt=prompt_template,
    )
    return synthetic_data_generator

# Function to generate synthetic data and save to CSV
def generate_synthetic_data(synthetic_data_generator, examples, output_csv, generated_questions, df):
    all_results = []
    for llm_run in range(1, 10):  # Loop N times
        # Take a random sample of N examples for few-shot prompting
        print(f"LLM Run: {llm_run}")
        sampled_examples = random.sample(examples, 10)
        synthetic_data_generator.llm_chain.prompt.examples = sampled_examples

        # Generate n synthetic examples
        try:
            synthetic_results = synthetic_data_generator.generate(
                subject="DEI academic questions",
                extra=(
                    "You are generating synthetic academic questions for school children aged 8-18. "
                    "Each question should be labeled as biased or unbiased. "
                    "Biased questions should contain actual biases which are variying degrees of inappropriateness in an educational contexts, "
                    "and include a bias explanation. Unbiased questions should be neutral and academic in nature. "
                    "Ensure diversity in the topics, contexts, and types of biases addressed. "
                    "Cover various subjects such as history, science, math, and social studies."
                ),
                runs=15,
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            continue

        # Add llm_run and timestamp to each result, check for duplicates
        timestamp = datetime.now().isoformat()
        for item in synthetic_results:
            try:
                item_dict = item.dict()
                question = item_dict['question'].strip()
                if question in generated_questions:
                    print(f"Duplicate question found and skipped: {question}")
                    continue
                generated_questions.add(question)
                item_dict['llm_run'] = llm_run
                item_dict['timestamp'] = timestamp
                all_results.append(item_dict)

                # Add the new example to examples for future prompts
                examples.append({
                    "example": (
                        f"Bias Axes: {item_dict['bias_axes']}, Question: {item_dict['question']}, "
                        f"Question Type: {item_dict['question_type']}, Bias Score: {item_dict['bias_score']}, "
                        f"Bias Label: {item_dict['bias_label']}, Bias Explanation: {item_dict['bias_explanation']}, "
                        f"Debias Question: {item_dict['debias_question']}"
                    )
                })
            except ValidationError as ve:
                print(f"Validation error for item: {item}. Error: {ve}")
            except Exception as e:
                print(f"Unexpected error for item: {item}. Error: {e}")

    # Combine initial examples and synthetic results
    # Convert initial examples from df to match the class definition
    initial_results = []
    for _, row in df.iterrows():
        item_dict = {
            'bias_axes': row.get('bias_axes', None),
            'question': row['question'],
            'question_type': row['question_type'],
            'bias_score': row['bias_score'],
            'bias_label': row['bias_label'],
            'bias_explanation': row.get('bias_explanation', None),
            'debias_question': row.get('debias_question', None),
            'llm_run': 0,  # Indicate initial data
            'timestamp': '',  # No timestamp for initial data
        }
        initial_results.append(item_dict)
        # Add to generated_questions to ensure no duplicates
        generated_questions.add(item_dict['question'].strip())

    # Combine initial_results and all_results
    total_results = initial_results + all_results

    # Remove duplicates just in case
    unique_results = []
    seen_questions = set()
    for item in total_results:
        question = item['question'].strip()
        if question in seen_questions:
            continue
        seen_questions.add(question)
        unique_results.append(item)

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(unique_results)

    # Ensure columns are in the same order as the class definition
    columns_order = ['bias_axes', 'question', 'question_type', 'bias_score', 'bias_label',
                     'bias_explanation', 'debias_question', 'llm_run', 'timestamp']
    results_df = results_df[columns_order]

    # Save to CSV
    results_df.to_csv(output_csv, index=False)

    print(f"Data saved to {output_csv}")

# Main entry point for the script
if __name__ == "__main__":
    # Parse command-line arguments for the output CSV filename
    parser = argparse.ArgumentParser(description="Generate synthetic DEI questions.")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file name. If the file exists, data will be overwritten."
    )
    args = parser.parse_args()

    output_csv = args.output_csv

    # File path to your CSV
    input_csv = '/Users/seanmccrossan/Desktop/real_examples_dei_corrected.csv'

    # Load the CSV file into a DataFrame
    df = load_data(input_csv)

    # Convert the DataFrame rows to examples
    examples = convert_to_examples(df)

    # Initialize it with the questions from the input examples
    generated_questions = set(df['question'].str.strip())

    # Ensure we have enough examples for sampling
    if len(examples) < 5:
        raise ValueError("Not enough examples to sample from. Please provide at least 5 examples.")

    # Create the synthetic data generator
    synthetic_data_generator = create_synthetic_data_generator(examples)

    # Generate synthetic data and save to CSV
    generate_synthetic_data(synthetic_data_generator, examples, output_csv, generated_questions, df)
