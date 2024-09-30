from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import pandas as pd

# Load your CSV file into a DataFrame
file_path = '/Users/seanmccrossan/Desktop/real_examples_dei_corrected.csv'
df = pd.read_csv(file_path)

# Define the schema for synthetic data
class DEIQuestions(BaseModel):
    bias_axes: str
    question: str
    question_type: str
    bias_score: float
    bias_label: bool
    bias_explanation: str
    debias_question: str


# Convert DataFrame rows to examples
examples = []
for _, row in df.iterrows():
    examples.append({
        "example": f"""Bias Axes: {row['bias_axes']}, Question: {row['question']}, 
        Question Type: {row['question_type']}, Bias Score: {row['bias_score']}, 
        Bias Label: {row['bias_label']}, Bias Explanation: {row['bias_explanation']}, 
        Debias Question: {row['debias_question']}"""
    })

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
    llm=ChatOpenAI(
        temperature=1
    ),  # You'll need to replace with your actual Language Model instance
    prompt=prompt_template,
)

# Generate synthetic data
synthetic_results = synthetic_data_generator.generate(
    subject="DEI_questions",
    extra="Generate diverse examples with unique bias explanations. Make sure questions span different contexts.",
    runs=10,
)

# Display the synthetic data
for result in synthetic_results:
    print(result.json())