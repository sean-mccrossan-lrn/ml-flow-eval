# main.py

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import re
import json
import openai
import pandas as pd
import wandb
import numpy as np
import asyncio
import weave
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Import custom modules
from Model.models import BiasModel
from Metrics.agent_metrics import AgentMetrics
from utils.s3_manager import S3Manager

# Configuration Parameters
CONFIG = {
    "weave_project": "bias-detection-evaluation-demo-new",
    "openai_api_key_env": "OPENAI_API_KEY",
    "model_config_path": "bias_detection_wb/model_config.json",
    "prompts_path": "bias_detection_wb/prompts.json",
    "s3": {
        "bucket_name": "learnosity-ds-datasets",
        "object_name": "data/bias/generated_synthetic_data.csv",
        "region": "us-east-1"
    },
    "weave_dataset_name": "bias_detection_dataset_new",
    "wandb": {
        "project": "bias-detection-evaluation-demo-new",
        "group": "bias_evaluation",
        "job_type": "evaluation"
    },
    "column_names": {
        "bias_axes": "bias_axes",
        "question": "question",
        "question_type": "question_type",
        "bias_score": "bias_score",
        "bias_label": "bias_label",
        "bias_explanation": "bias_explanation",
        "debias_question": "debias_question",
        "llm_run": "llm_run",
        "timestamp": "timestamp",
        # Additional columns
        "predicted_label": "predicted_label",
        "debiased_text": "debiased_text"
    }
}

def add_parent_directory_to_sys_path():
    """
    Adds the parent directory to sys.path to allow module imports.
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)

def initialize_weave():
    """
    Initializes the Weave project.
    """
    weave.init(CONFIG["weave_project"])

def get_openai_api_key():
    """
    Retrieves the OpenAI API key from environment variables.
    """
    api_key = os.getenv(CONFIG["openai_api_key_env"])
    if not api_key:
        print(f"Please set your OpenAI API key in the environment variable '{CONFIG['openai_api_key_env']}'.")
        sys.exit(1)
    openai.api_key = api_key

def load_json_file(file_path):
    """
    Loads a JSON file and returns its content.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File at path '{file_path}' is not a valid JSON.")
        sys.exit(1)

def load_dataset():
    """
    Loads the dataset from S3, processes it, and takes a random sample of 5 rows.
    """
    s3_manager = S3Manager(
        bucket_name=CONFIG["s3"]["bucket_name"],
        region=CONFIG["s3"]["region"]
    )
    
    df = s3_manager.download_file_to_dataframe(CONFIG["s3"]["object_name"])
    if df is None:
        print("Failed to download the dataset from S3. Please check the bucket name and object key.")
        sys.exit(1)
    
    # Check if required columns are present
    check_required_columns(df)
    
    # Handle NaN values in 'debias_question'
    df[CONFIG["column_names"]["debias_question"]].fillna('', inplace=True)
    
    # Take a random sample of 5 rows for testing purposes
    df = df.sample(n=3, random_state=42)
    
    return df


def check_required_columns(df):
    """
    Checks if the required columns are present in the DataFrame.
    """
    required_columns = [
        CONFIG["column_names"]["bias_axes"],
        CONFIG["column_names"]["question"],
        CONFIG["column_names"]["question_type"],
        CONFIG["column_names"]["bias_score"],
        CONFIG["column_names"]["bias_label"],
        CONFIG["column_names"]["bias_explanation"],
        CONFIG["column_names"]["debias_question"],
        CONFIG["column_names"]["llm_run"],
        CONFIG["column_names"]["timestamp"]
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing from the dataset: {missing_columns}")
        sys.exit(1)

def publish_weave_dataset(examples):
    """
    Publishes the dataset to Weave.
    """
    dataset = weave.Dataset(name=CONFIG["weave_dataset_name"], rows=examples)
    weave.publish(dataset)
    return dataset

def create_dataset_mapping(examples):
    """
    Creates a mapping from question to debiased question.
    """
    return {example[CONFIG["column_names"]["question"]]: example[CONFIG["column_names"]["debias_question"]] for example in examples}

def initialize_wandb_run(model_name, prompt_template):
    """
    Initializes a W&B run for the given model.
    """
    wandb.init(
        project=CONFIG["wandb"]["project"],
        name=f"evaluation_{model_name}",
        group=CONFIG["wandb"]["group"],
        job_type=CONFIG["wandb"]["job_type"],
        config={
            "model_name": model_name,
            "prompt_template": prompt_template
        }
    )

def log_confusion_matrix(model_name, true_labels, predicted_labels):
    """
    Calculates and logs the confusion matrix to W&B.
    """
    label_mapping = {False: 0, True: 1}
    true_labels_int = [label_mapping[label] for label in true_labels]
    predicted_labels_int = [label_mapping[label] for label in predicted_labels]
    
    cm = confusion_matrix(true_labels_int, predicted_labels_int, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['unbiased', 'biased'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    
    wandb.log({"confusion_matrix_plot": wandb.Image(plt)})
    plt.clf()

def evaluate_model(model_name, prompts, examples, dataset):
    """
    Evaluates a single model and logs the results.
    """
    # Start W&B run
    initialize_wandb_run(model_name, prompts['bias_analysis']['prompt'])
    
    print(f"Evaluating model: {model_name}")
    
    # Instantiate the BiasModel
    debiaser = BiasModel(
        model_name=model_name,
        prompt_template=prompts['bias_analysis']['prompt']
    )
    
    # Initialize evaluation metrics
    true_labels = []
    predicted_labels = []
    results = []
    
    for example in examples:
        label = example[CONFIG["column_names"]["bias_label"]]
        question = example[CONFIG["column_names"]["question"]]
        model_output = debiaser.predict(question)
        
        if model_output is None:
            predicted_label = False
            bias_score = None
            bias_axes = None
            debiased_text = None
        else:
            predicted_label = True if model_output.get('bias_score', 0) > 0 else False
            bias_score = model_output.get('bias_score', None)
            bias_axes = model_output.get('bias_axes', None)
            debiased_text = model_output.get('debiased_text', None)
        
        true_labels.append(label)
        predicted_labels.append(predicted_label)
        
        # Update example with predictions
        example[CONFIG["column_names"]["predicted_label"]] = predicted_label
        example[CONFIG["column_names"]["bias_score"]] = bias_score
        example[CONFIG["column_names"]["bias_axes"]] = bias_axes
        example[CONFIG["column_names"]["debiased_text"]] = debiased_text
        
        # Include 'debias_question' and 'question' in model_output
        if model_output is not None:
            model_output[CONFIG["column_names"]["debias_question"]] = example[CONFIG["column_names"]["debias_question"]]
            model_output[CONFIG["column_names"]["question"]] = example[CONFIG["column_names"]["question"]]
        else:
            model_output = {
                CONFIG["column_names"]["debias_question"]: example[CONFIG["column_names"]["debias_question"]],
                CONFIG["column_names"]["question"]: example[CONFIG["column_names"]["question"]],
                CONFIG["column_names"]["debiased_text"]: None,
                CONFIG["column_names"]["bias_score"]: None,
                CONFIG["column_names"]["bias_axes"]: None
            }
        
        # Debugging statements
        print(f"Model Output for Question: {model_output[CONFIG['column_names']['question']]}")
        print(f"Debiased Text: {model_output.get(CONFIG['column_names']['debiased_text'], None)}")
        print(f"Debiased Question (Ground Truth): {model_output[CONFIG['column_names']['debias_question']]}")
        
        results.append(example)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, pos_label=True, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, pos_label=True, zero_division=0)
    
    # Print evaluation results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # Log confusion matrix
    log_confusion_matrix(model_name, true_labels, predicted_labels)
    
    # Log evaluation metrics to W&B
    wandb.log({
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1
    })
    
    # Finish W&B run
    wandb.finish()
    
    # Print results DataFrame
    results_df = pd.DataFrame(results)
    print(results_df[[CONFIG["column_names"]["question"], CONFIG["column_names"]["bias_label"], CONFIG["column_names"]["predicted_label"]]])
    
    # Perform Weave Evaluation with AgentMetrics
    perform_weave_evaluation_with_agent_metrics(model_name, examples, dataset, debiaser)
    
    # Perform Weave Evaluation with custom scorer
    perform_weave_evaluation(model_name, dataset, debiaser)

def perform_weave_evaluation_with_agent_metrics(model_name, examples, dataset, debiaser):
    """
    Performs Weave evaluation using AgentMetrics scorer.
    """
    # Instantiate the AgentMetrics scorer
    agent_metrics_scorer = AgentMetrics.from_dataset(
        model_name='gpt-4o',
        openai_api_key=openai.api_key,
        dataset=examples
    )
    
    # Define the evaluation
    evaluation = weave.Evaluation(
        name=f'agent_metrics_evaluation_{model_name}',
        dataset=dataset,
        scorers=[agent_metrics_scorer],
    )
    
    # Run the evaluation asynchronously
    try:
        weave_results = asyncio.run(evaluation.evaluate(debiaser))
    except Exception as e:
        print(f"Error during Weave evaluation: {e}")
        return
    
    # Check if 'examples' key exists in results
    if 'examples' not in weave_results:
        print(f"Evaluation did not produce any examples. Results: {weave_results}")
        return
    
    weave_examples_results = weave_results['examples']
    
    # Extract metrics from Weave results
    metrics = ["answer_relevancy", "correctness", "hallucination", "contextual_relevancy"]
    metric_results = {metric: [] for metric in metrics}
    
    for example_result in weave_examples_results:
        scores = example_result['scores']['AgentMetrics']
        for metric in metrics:
            metric_results[metric].append(scores.get(metric))
    
    # Optionally, print per-example results from Weave evaluation
    weave_results_df = pd.DataFrame(weave_examples_results)
    print(weave_results_df[[f'input.{CONFIG["column_names"]["question"]}', 
                            f'input.{CONFIG["column_names"]["bias_label"]}', 
                            'scores.AgentMetrics']])

def perform_weave_evaluation(model_name, dataset, debiaser):
    """
    Performs Weave evaluation using a custom bias_scorer.
    """
    # Define the scorer function
    @weave.op()
    def bias_scorer(bias_label: bool, model_output: dict) -> dict:
        predicted_label = False if model_output is None else (True if model_output.get('bias_score', 0) > 0 else False)
        is_correct = predicted_label == bias_label
        return {'correct': is_correct, 'predicted_label': predicted_label}
    
    # Define the evaluation
    evaluation = weave.Evaluation(
        name=f'bias_evaluation_{model_name}',
        dataset=dataset,
        scorers=[bias_scorer],
    )
    
    # Run the evaluation asynchronously
    weave_results = asyncio.run(evaluation.evaluate(debiaser))
    
    # Check if 'examples' key exists in results
    if 'examples' not in weave_results:
        print(f"Evaluation did not produce any examples. Results: {weave_results}")
        return
    
    weave_examples_results = weave_results['examples']
    
    # Extract predicted labels and true labels from Weave results
    weave_predicted_labels = []
    weave_true_labels = []
    
    for example_result in weave_examples_results:
        input_data = example_result['input']
        score = example_result['scores']['bias_scorer']
        
        weave_true_labels.append(input_data[CONFIG["column_names"]["bias_label"]])
        weave_predicted_labels.append(score['predicted_label'])
    
    # Calculate and print evaluation metrics from Weave results
    weave_accuracy = accuracy_score(weave_true_labels, weave_predicted_labels)
    weave_recall = recall_score(weave_true_labels, weave_predicted_labels, pos_label=True, zero_division=0)
    weave_f1 = f1_score(weave_true_labels, weave_predicted_labels, pos_label=True, zero_division=0)
    
    print(f"Weave Evaluation for Model: {model_name}")
    print(f"Weave Accuracy: {weave_accuracy:.2f}")
    print(f"Weave Recall: {weave_recall:.2f}")
    print(f"Weave F1-Score: {weave_f1:.2f}")
    
    # Optionally, print per-example results from Weave evaluation
    weave_results_df = pd.DataFrame(weave_examples_results)
    print(weave_results_df[[f'input.{CONFIG["column_names"]["question"]}', 
                            f'input.{CONFIG["column_names"]["bias_label"]}', 
                            'scores.bias_scorer.predicted_label']])

def main():
    """
    Main function to execute the bias detection evaluation.
    """
    # Add parent directory to sys.path for module imports
    add_parent_directory_to_sys_path()
    
    # Initialize Weave project
    initialize_weave()
    
    # Set OpenAI API key
    get_openai_api_key()
    
    # Load configuration and prompts
    config = load_json_file(CONFIG["model_config_path"])
    models = config.get('models', [])
    
    prompts = load_json_file(CONFIG["prompts_path"])
    
    # Load and process dataset
    df = load_dataset()
    
    # Convert DataFrame to list of examples
    examples = df.to_dict(orient='records')
    
    # Publish dataset to Weave
    dataset = publish_weave_dataset(examples)
    
    # Create dataset mapping (if needed elsewhere)
    dataset_mapping = create_dataset_mapping(examples)
    
    # Iterate over all OpenAI models and evaluate
    for model_name in models:
        evaluate_model(model_name, prompts, examples, dataset)

if __name__ == "__main__":
    main()
