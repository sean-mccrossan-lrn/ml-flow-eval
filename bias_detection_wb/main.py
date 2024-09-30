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

# Import BiasModel from the Model.models module
from Model.models import BiasModel
from Metrics.agent_metrics import AgentMetrics
from utils.s3_manager import S3Manager

def main():
    # Initialize Weave project
    weave.init('bias-detection-evaluation-demo')

    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")
        return

    # Load config
    with open('bias_detection_wb/model_config.json', 'r') as config_file:
        config = json.load(config_file)
        models = config['models']

    # Load prompts
    with open('bias_detection_wb/prompts.json', 'r') as prompt_file:
        prompts = json.load(prompt_file)

    # Load dataset from S3 using S3Manager
    bucket_name = 'learnosity-ds-datasets'
    object_name = 'data/bias/Synthetic_Complete_School_Homework_Questions_Dataset.csv'

    s3_manager = S3Manager(bucket_name, region='us-east-1')


    # Download and read the CSV into a DataFrame
    df = s3_manager.download_file_to_dataframe(object_name)
    if df is None:
        print(f"Failed to download the dataset from S3. Please check the bucket name and object key.")
        return

    # **Rename columns to ensure consistent naming**
    df.rename(columns={'Debiased Question':'Debiased_Question'}, inplace=True)

    # Create 'label' column based on 'Bias Score'
    try:
        df['label'] = df['Bias Score'].apply(lambda x: 'biased' if x > 0 else 'unbiased')
    except KeyError:
        print("Error: 'Bias Score' column not found in the dataset.")
        return
    except Exception as e:
        print(f"Unexpected error while processing the DataFrame: {e}")
        return

    # Convert DataFrame to list of examples
    examples = df.to_dict(orient='records')

    # Create and publish dataset to Weave
    dataset = weave.Dataset(name='bias_detection_dataset', rows=examples)
    weave.publish(dataset)

    # **Create the dataset mapping from Question to Debiased_Question**
    dataset_mapping = {example['Question']: example['Debiased_Question'] for example in examples}

    # Iterate over all OpenAI models
    for model_name in models:
        # Start a new W&B run for each model
        wandb.init(
            project="bias-detection-evaluation-demo",
            name=f"evaluation_{model_name}",
            group="bias_evaluation",
            job_type="evaluation",
            config={
                "model_name": model_name,
                "prompt_template": prompts['bias_analysis']['prompt']
            }
        )

        print(f"Evaluating model: {model_name}")

        # Instantiate the BiasModel for each model
        debiaser = BiasModel(
            model_name=model_name,
            prompt_template=prompts['bias_analysis']['prompt']
        )

        # === Original Evaluation Logic ===

        # Initialize lists to collect true labels and predictions
        true_labels = []
        predicted_labels = []
        results = []

        for example in examples:
            label = example['label']
            question = example['Question']
            model_output = debiaser.predict(question)
            if model_output is None:
                predicted_label = 'unbiased'
                bias_score = None
                bias_axes = None
                debiased_text = None
            else:
                predicted_label = 'biased' if model_output.get('bias_score', 0) > 0 else 'unbiased'
                bias_score = model_output.get('bias_score', None)
                bias_axes = model_output.get('bias_axes', None)
                debiased_text = model_output.get('debiased_text', None)

            true_labels.append(label)
            predicted_labels.append(predicted_label)
            # Collect additional data
            example['predicted_label'] = predicted_label
            example['bias_score'] = bias_score
            example['bias_axes'] = bias_axes
            example['debiased_text'] = debiased_text

            # **Include 'Debiased_Question' and 'Question' in model_output**
            if model_output is not None:
                model_output['Debiased_Question'] = example['Debiased_Question']
                model_output['Question'] = example['Question']
            else:
                model_output = {
                    'Debiased_Question': example['Debiased_Question'],
                    'Question': example['Question'],
                    'debiased_text': None,
                    'bias_score': None,
                    'bias_axes': None
                }

            # **Add Debugging Statements**
            print(f"Model Output for Question: {model_output['Question']}")
            print(f"Debiased Text: {model_output['debiased_text']}")
            print(f"Debiased Question (Ground Truth): {model_output['Debiased_Question']}")

            results.append(example)


        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels, pos_label='biased', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, pos_label='biased', zero_division=0)

        # Print the evaluation results
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        # Map labels to integers for confusion matrix
        label_mapping = {'unbiased': 0, 'biased': 1}
        true_labels_int = [label_mapping[label] for label in true_labels]
        predicted_labels_int = [label_mapping[label] for label in predicted_labels]

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels_int, predicted_labels_int, labels=[0, 1])

        # Create confusion matrix display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['unbiased', 'biased'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')

        # Log the confusion matrix plot to W&B
        wandb.log({"confusion_matrix_plot": wandb.Image(plt)})

        # Clear the plot for the next model
        plt.clf()

        # Log evaluation metrics to W&B
        wandb.log({
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1
        })

        # Finish the W&B run
        wandb.finish()

        # Convert results to DataFrame and print
        results_df = pd.DataFrame(results)
        print(results_df[['Question', 'label', 'predicted_label']])

        # === Weave Evaluation Logic with AgentMetrics ===

        # **Instantiate the AgentMetrics scorer with dataset mapping**
        agent_metrics_scorer = AgentMetrics.from_dataset(
            model_name='gpt-4', 
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
            continue  

        # Extract per-example results
        if 'examples' not in weave_results:
            print(f"Evaluation did not produce any examples. Results: {weave_results}")
            continue  # Skip to the next model or handle as needed

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
        print(weave_results_df[['input.Question', 'input.label', 'scores.AgentMetrics']])

        # === Weave Evaluation Logic ===

        # Define the scorer function
        @weave.op()
        def bias_scorer(label: str, model_output: dict) -> dict:
            if model_output is None:
                predicted_label = 'unbiased'
            else:
                predicted_label = 'biased' if model_output.get('bias_score', 0) > 0 else 'unbiased'
            is_correct = predicted_label == label
            return {'correct': is_correct, 'predicted_label': predicted_label}

        # Define the evaluation
        evaluation = weave.Evaluation(
            name=f'bias_evaluation_{model_name}',
            dataset=dataset,
            scorers=[bias_scorer],
        )

        # Run the evaluation asynchronously
        weave_results = asyncio.run(evaluation.evaluate(debiaser))

        # Extract per-example results
        if 'examples' not in weave_results:
            print(f"Evaluation did not produce any examples. Results: {weave_results}")
            continue  # Skip to the next model or handle as needed

        weave_examples_results = weave_results['examples']

        # Extract predicted labels and true labels from Weave results
        weave_predicted_labels = []
        weave_true_labels = []

        for example_result in weave_examples_results:
            input_data = example_result['input']
            model_output = example_result['model_output']
            score = example_result['scores']['bias_scorer']

            weave_true_labels.append(input_data['label'])
            weave_predicted_labels.append(score['predicted_label'])

        # Optionally, calculate and print evaluation metrics from Weave results
        weave_accuracy = accuracy_score(weave_true_labels, weave_predicted_labels)
        weave_recall = recall_score(weave_true_labels, weave_predicted_labels, pos_label='biased', zero_division=0)
        weave_f1 = f1_score(weave_true_labels, weave_predicted_labels, pos_label='biased', zero_division=0)

        print(f"Weave Evaluation for Model: {model_name}")
        print(f"Weave Accuracy: {weave_accuracy:.2f}")
        print(f"Weave Recall: {weave_recall:.2f}")
        print(f"Weave F1-Score: {weave_f1:.2f}")

        # Optionally, print per-example results from Weave evaluation
        weave_results_df = pd.DataFrame(weave_examples_results)
        print(weave_results_df[['input.Question', 'input.label', 'scores.bias_scorer.predicted_label']])

def json_main():

    # Initialize Weave project
    weave.init('bias-detection-evaluation-json-demo')

    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")
        return

    # Load config
    with open('bias_detection_wb/model_config.json', 'r') as config_file:
        config = json.load(config_file)
        models = config['models']

    # Load prompts
    with open('bias_detection_wb/prompts.json', 'r') as prompt_file:
        prompts = json.load(prompt_file)
        # Access the specific JSON prompt
        json_prompt = prompts['bias_analyses_json_prompt']['prompt']

    bucket_name = 'learnosity-ds-datasets'

    upload_object_name = 'data/bias/synthetic_bias_json.xlsx'

    # Initialize S3Manager
    s3_manager = S3Manager(bucket_name, region='us-east-1')

    # Debug: List objects in the bucket
    print("Listing objects in the bucket...")
    s3_manager.list_objects()

    # Download and read the CSV into a DataFrame
    df = s3_manager.download_excel_file_to_dataframe(upload_object_name,header=1)
    if df is None:
        print(f"Failed to download the dataset from S3. Please check the bucket name and object key.")

    # **Rename columns to ensure consistent naming**
    df.rename(columns={'Debiased Question':'Debiased_Question'}, inplace=True)

    # Create 'label' column based on 'Bias Score'
    try:
        df['label'] = df['Bias Score'].apply(lambda x: 'biased' if x > 0 else 'unbiased')
    except KeyError:
        print("Error: 'Bias Score' column not found in the dataset.")
    except Exception as e:
        print(f"Unexpected error while processing the DataFrame: {e}")

    print(df)

    # Convert DataFrame to list of examples
    examples = df.to_dict(orient='records')

    # Create and publish dataset to Weave
    dataset = weave.Dataset(name='bias_detection_dataset', rows=examples)
    weave.publish(dataset)

    # **Create the dataset mapping from Question to Debiased_Question**
    dataset_mapping = {example['Question']: example['Debiased_Question'] for example in examples}

    # Iterate over all OpenAI models
    for model_name in models:
        # Start a new W&B run for each model
        wandb.init(
            project="bias-detection-evaluation-json-demo",
            name=f"evaluation_{model_name}",
            group="bias_evaluation",
            job_type="evaluation",
            config={
                "model_name": model_name,
                "prompt_template": prompts['bias_analysis']['prompt']
            }
        )

        print(f"Evaluating model: {model_name}")

        # Instantiate the BiasModel for each model
        debiaser = BiasModel(
            model_name=model_name,
            prompt_template=prompts['bias_analysis']['prompt']
        )

        # === Original Evaluation Logic ===

        # Initialize lists to collect true labels and predictions
        true_labels = []
        predicted_labels = []
        results = []

        for example in examples:
            label = example['label']
            question = example['Question']
            model_output = debiaser.predict(question)
            if model_output is None:
                predicted_label = 'unbiased'
                bias_score = None
                bias_axes = None
                debiased_text = None
            else:
                predicted_label = 'biased' if model_output.get('bias_score', 0) > 0 else 'unbiased'
                bias_score = model_output.get('bias_score', None)
                bias_axes = model_output.get('bias_axes', None)
                debiased_text = model_output.get('debiased_text', None)

            true_labels.append(label)
            predicted_labels.append(predicted_label)
            # Collect additional data
            example['predicted_label'] = predicted_label
            example['bias_score'] = bias_score
            example['bias_axes'] = bias_axes
            example['debiased_text'] = debiased_text

            # **Include 'Debiased_Question' and 'Question' in model_output**
            if model_output is not None:
                model_output['Debiased_Question'] = example['Debiased_Question']
                model_output['Question'] = example['Question']
            else:
                model_output = {
                    'Debiased_Question': example['Debiased_Question'],
                    'Question': example['Question'],
                    'debiased_text': None,
                    'bias_score': None,
                    'bias_axes': None
                }

            # **Add Debugging Statements**
            print(f"Model Output for Question: {model_output['Question']}")
            print(f"Debiased Text: {model_output['debiased_text']}")
            print(f"Debiased Question (Ground Truth): {model_output['Debiased_Question']}")

            results.append(example)


        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels, pos_label='biased', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, pos_label='biased', zero_division=0)

        # Print the evaluation results
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        # Map labels to integers for confusion matrix
        label_mapping = {'unbiased': 0, 'biased': 1}
        true_labels_int = [label_mapping[label] for label in true_labels]
        predicted_labels_int = [label_mapping[label] for label in predicted_labels]

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels_int, predicted_labels_int, labels=[0, 1])

        # Create confusion matrix display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['unbiased', 'biased'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')

        # Log the confusion matrix plot to W&B
        wandb.log({"confusion_matrix_plot": wandb.Image(plt)})

        # Clear the plot for the next model
        plt.clf()

        # Log evaluation metrics to W&B
        wandb.log({
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1
        })

        # Finish the W&B run
        wandb.finish()

        # Convert results to DataFrame and print
        results_df = pd.DataFrame(results)
        print(results_df[['Question', 'label', 'predicted_label']])

        # === Weave Evaluation Logic with AgentMetrics ===

        # **Instantiate the AgentMetrics scorer with dataset mapping**
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
            continue  

        # Extract per-example results
        if 'examples' not in weave_results:
            print(f"Evaluation did not produce any examples. Results: {weave_results}")
            continue  # Skip to the next model or handle as needed

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
        print(weave_results_df[['input.Question', 'input.label', 'scores.AgentMetrics']])

        # === Weave Evaluation Logic ===

        # Define the scorer function
        @weave.op()
        def bias_scorer(label: str, model_output: dict) -> dict:
            if model_output is None:
                predicted_label = 'unbiased'
            else:
                predicted_label = 'biased' if model_output.get('bias_score', 0) > 0 else 'unbiased'
            is_correct = predicted_label == label
            return {'correct': is_correct, 'predicted_label': predicted_label}

        # Define the evaluation
        evaluation = weave.Evaluation(
            name=f'bias_evaluation_{model_name}',
            dataset=dataset,
            scorers=[bias_scorer],
        )

        # Run the evaluation asynchronously
        weave_results = asyncio.run(evaluation.evaluate(debiaser))

        # Extract per-example results
        if 'examples' not in weave_results:
            print(f"Evaluation did not produce any examples. Results: {weave_results}")
            continue  # Skip to the next model or handle as needed

        weave_examples_results = weave_results['examples']

        # Extract predicted labels and true labels from Weave results
        weave_predicted_labels = []
        weave_true_labels = []

        for example_result in weave_examples_results:
            input_data = example_result['input']
            model_output = example_result['model_output']
            score = example_result['scores']['bias_scorer']

            weave_true_labels.append(input_data['label'])
            weave_predicted_labels.append(score['predicted_label'])

        # Optionally, calculate and print evaluation metrics from Weave results
        weave_accuracy = accuracy_score(weave_true_labels, weave_predicted_labels)
        weave_recall = recall_score(weave_true_labels, weave_predicted_labels, pos_label='biased', zero_division=0)
        weave_f1 = f1_score(weave_true_labels, weave_predicted_labels, pos_label='biased', zero_division=0)

        print(f"Weave Evaluation for Model: {model_name}")
        print(f"Weave Accuracy: {weave_accuracy:.2f}")
        print(f"Weave Recall: {weave_recall:.2f}")
        print(f"Weave F1-Score: {weave_f1:.2f}")

        # Optionally, print per-example results from Weave evaluation
        weave_results_df = pd.DataFrame(weave_examples_results)
        print(weave_results_df[['input.Question', 'input.label', 'scores.bias_scorer.predicted_label']])

if __name__ == "__main__":
    json_main()
