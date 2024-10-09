# main.py

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import json
import openai
import pandas as pd
import wandb
import asyncio
import weave
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Import custom modules
from Model.models import BiasModel
from Metrics.agent_metrics import AgentMetrics, OpenAIModelProvider 
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
        "region": "us-east-1",
        "num_rows": 10
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
        "predicted_debias_question": "predicted_debias_question",
        "predicted_bias_score": "predicted_bias_score",
        "predicted_bias_axes": "predicted_bias_axes",
        "predicted_bias_explanation": "predicted_bias_explanation"
    },
    # AgentMetrics Configuration
    "agent_metrics": {
        "models": [
            {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "num_runs": 1
            },
            {
                "provider": "openai",
                "model_name": "gpt-4o",
                "num_runs": 1
            }
        ]
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
    Loads the dataset from S3, processes it, and takes a sample of rows.
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
    
    # Take a random sample of rows for testing purposes
    df = df.sample(n=CONFIG["s3"]["num_rows"], random_state=42)
    
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

def initialize_wandb_run(model_name, provider_name, prompt_template):
    """
    Initializes a W&B run for the given model, provider, and prompt.
    """
    wandb.init(
        project=CONFIG["wandb"]["project"],
        name=f"evaluation_{model_name}_{provider_name}",
        group=CONFIG["wandb"]["group"],
        job_type=CONFIG["wandb"]["job_type"],
        config={
            "model_name": model_name,
            "provider_name": provider_name,
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
    
    # Convert plot to W&B Image
    wandb.log({"confusion_matrix_plot": wandb.Image(plt)})
    plt.clf()

def evaluate_model(model_name, provider_name, prompt_template, prompts, examples, dataset):
    """
    Evaluates a single model-provider-prompt combination and logs the results.
    """
    # Start W&B run with model, provider, and prompt details
    initialize_wandb_run(model_name, provider_name, prompt_template)
    
    print(f"Evaluating model: {model_name} with provider: {provider_name}")
    
    # Instantiate the BiasModel
    debiaser = BiasModel(
        model_name=model_name,
        prompt_template=prompt_template
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
            predicted_bias_score = 0.0
            predicted_bias_axes = []
            predicted_debias_question = ""
            predicted_bias_explanation = ""
        else:
            predicted_label = True if model_output.get('predicted_bias_score', 0) > 0 else False
            predicted_bias_score = model_output.get('predicted_bias_score', 0.0)
            predicted_bias_axes = model_output.get('predicted_bias_axes', [])
            predicted_debias_question = model_output.get('predicted_debias_question', "")
            predicted_bias_explanation = model_output.get('predicted_bias_explanation', "")
        
        true_labels.append(label)
        predicted_labels.append(predicted_label)
        
        # Update example with predictions
        example[CONFIG["column_names"]["predicted_label"]] = predicted_label
        example[CONFIG["column_names"]["predicted_bias_score"]] = predicted_bias_score
        example[CONFIG["column_names"]["predicted_bias_axes"]] = predicted_bias_axes
        example[CONFIG["column_names"]["predicted_debias_question"]] = predicted_debias_question
        example[CONFIG["column_names"]["predicted_bias_explanation"]] = predicted_bias_explanation
        
        # Include 'debias_question' and 'question' in model_output
        if model_output is not None:
            model_output[CONFIG["column_names"]["debias_question"]] = example[CONFIG["column_names"]["debias_question"]]
            model_output[CONFIG["column_names"]["question"]] = example[CONFIG["column_names"]["question"]]
        else:
            model_output = {
                CONFIG["column_names"]["debias_question"]: example[CONFIG["column_names"]["debias_question"]],
                CONFIG["column_names"]["question"]: example[CONFIG["column_names"]["question"]],
                CONFIG["column_names"]["predicted_debias_question"]: "",
                CONFIG["column_names"]["predicted_bias_score"]: 0.0,
                CONFIG["column_names"]["predicted_bias_axes"]: [],
                CONFIG["column_names"]["predicted_bias_explanation"]: ""
            }
        
        # Debugging statements
        print(f"Model Output for Question: {model_output[CONFIG['column_names']['question']]}")
        print(f"Predicted Debiased Text: {model_output.get(CONFIG['column_names']['predicted_debias_question'], None)}")
        print(f"Debiased Question (Ground Truth): {model_output[CONFIG['column_names']['debias_question']]}")
        
        results.append(example)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, pos_label=True, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, pos_label=True, zero_division=0)
    
    # Perform Weave Evaluation with AgentMetrics
    agent_metrics_results = perform_weave_evaluation_with_agent_metrics(
        model_name, provider_name, prompts, examples, dataset, debiaser
    )
    
    # Perform Weave Evaluation with custom scorer (Bias Scorer)
    weave_evaluation_results = perform_weave_evaluation(
        model_name, provider_name, prompts, dataset, debiaser
    )
    
    # Ensure that the results are dictionaries
    agent_metrics_results = agent_metrics_results if isinstance(agent_metrics_results, dict) else {}
    weave_evaluation_results = weave_evaluation_results if isinstance(weave_evaluation_results, dict) else {}
    
    # Combine all metrics into a single dictionary
    metrics_to_log = {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1,
        **agent_metrics_results,    # Include AgentMetrics results
        **weave_evaluation_results  # Include Bias Scorer results
    }
    
    # Log all metrics to W&B
    wandb.log(metrics_to_log)
    
    # Log confusion matrix
    log_confusion_matrix(model_name, true_labels, predicted_labels)
    
    # Finish W&B run
    wandb.finish()
    
    # Print results DataFrame
    results_df = pd.DataFrame(results)
    print(results_df[[CONFIG["column_names"]["question"], CONFIG["column_names"]["bias_label"], CONFIG["column_names"]["predicted_label"]]])

def perform_weave_evaluation_with_agent_metrics(model_name, provider_name, prompts, examples, dataset, debiaser):
    """
    Performs Weave evaluation using AgentMetrics scorer and returns aggregated metrics.
    """
    # Initialize an empty dictionary to collect all agent metrics results
    all_agent_metrics_results = {}
    
    # Initialize model providers based on AgentMetrics configuration
    model_providers = []
    for model_info in CONFIG["agent_metrics"]["models"]:
        if model_info["provider"] == "openai":
            provider = OpenAIModelProvider(
                model_name=model_info["model_name"],
                openai_api_key=openai.api_key
            )
            model_providers.append(provider)
        else:
            print(f"Unsupported provider '{model_info['provider']}'. Skipping.")
    
    # Instantiate AgentMetrics scorer
    agent_metrics_scorer = AgentMetrics.from_dataset(
        model_providers=model_providers,
        dataset=examples,
        num_runs_per_provider={model_info["model_name"]: model_info["num_runs"] for model_info in CONFIG["agent_metrics"]["models"]},
        aggregation_method='mode'  # Options: 'mode', 'mean', 'median'
    )
    
    # Define the evaluation
    evaluation = weave.Evaluation(
        name=f'bias_evaluation_with_agent_metrics_{model_name}',
        dataset=dataset,
        scorers=[agent_metrics_scorer],
    )
    
    # Run the evaluation asynchronously
    weave_results = asyncio.run(evaluation.evaluate(debiaser))
    
    # Check if per-example results are available
    if 'examples' not in weave_results:
        print(f"AgentMetrics evaluation did not produce any examples. Results: {weave_results}")
        return all_agent_metrics_results  # Return empty if no results
    
    # Process per-example results
    examples_results = weave_results['examples']
    
    # Aggregate metrics
    total_counts = {
        'answer_relevancy': [],
        'correctness': [],
        'hallucination': [],
        'contextual_relevancy': []
    }

    for example_result in examples_results:
        scores = example_result['scores']['AgentMetrics']
        for provider_name_key, provider_scores in scores.items():
            for metric in total_counts.keys():
                value = provider_scores.get(metric)
                if value is not None:
                    total_counts[metric].append(value)
    
    # Calculate aggregated metrics based on the aggregation method
    for metric, values in total_counts.items():
        if values:
            try:
                aggregated_value = mode(values)
            except:
                aggregated_value = mean(values)  # Fallback to mean if no unique mode
            all_agent_metrics_results[f"agent_{metric}"] = aggregated_value
        else:
            all_agent_metrics_results[f"agent_{metric}"] = None
    
    return all_agent_metrics_results

def perform_weave_evaluation(model_name, provider_name, prompts, dataset, debiaser):
    """
    Performs Weave evaluation using a custom bias_scorer and returns the metrics.
    """
    # Define the scorer function
    @weave.op()
    def bias_scorer(bias_label: bool, model_output: dict) -> dict:
        predicted_label = False if model_output is None else (True if model_output.get('predicted_bias_score', 0) > 0 else False)
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
        print(f"Bias Scorer evaluation did not produce any examples. Results: {weave_results}")
        return {}
    
    weave_examples_results = weave_results['examples']

    # Extract predicted labels and true labels from Weave results
    weave_predicted_labels = []
    weave_true_labels = []

    for example_result in weave_examples_results:
        input_data = example_result['input']
        score = example_result['scores']['bias_scorer']

        weave_true_labels.append(input_data[CONFIG["column_names"]["bias_label"]])
        weave_predicted_labels.append(score['predicted_label'])

    # Calculate evaluation metrics from Weave results
    weave_accuracy = accuracy_score(weave_true_labels, weave_predicted_labels)
    weave_recall = recall_score(weave_true_labels, weave_predicted_labels, pos_label=True, zero_division=0)
    weave_f1 = f1_score(weave_true_labels, weave_predicted_labels, pos_label=True, zero_division=0)

    # Collect metrics for logging
    weave_metrics = {
        'bias_scorer_accuracy': weave_accuracy,
        'bias_scorer_recall': weave_recall,
        'bias_scorer_f1_score': weave_f1
    }

    return weave_metrics

def main():
    """
    Main function to execute the bias detection evaluation.
    """
    # # Add parent directory to sys.path for module imports
    # add_parent_directory_to_sys_path()
    
    # Initialize Weave project
    initialize_weave()
    
    # Set OpenAI API key
    get_openai_api_key()
    
    # Load configuration and prompts
    models_config = load_json_file(CONFIG["model_config_path"])
    prompts_config = load_json_file(CONFIG["prompts_path"])
    
    # Extract models and prompts
    models = models_config.get("models", [])
    prompts = prompts_config.get("bias_analysis", {})
    prompt_template = prompts.get("prompt", "")
    
    if not models:
        print("No models found in model_config.json.")
        sys.exit(1)
    
    if not prompt_template:
        print("No prompt found in prompts.json under 'bias_analysis'.")
        sys.exit(1)
    
    # Load and process dataset
    df = load_dataset()
    
    # Convert DataFrame to list of examples
    examples = df.to_dict(orient='records')
    
    # Publish dataset to Weave
    dataset = publish_weave_dataset(examples)
    
    # Create dataset mapping (if needed elsewhere)
    dataset_mapping = create_dataset_mapping(examples)
    
    # Iterate over all model and prompt combinations and evaluate
    for model_name in models:
        provider_name = "openai"  # Assuming all models in agent_metrics are from OpenAI
        evaluate_model(model_name, provider_name, prompt_template, prompts, examples, dataset)

if __name__ == "__main__":
    main()
