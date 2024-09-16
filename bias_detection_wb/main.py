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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Import BiasModel from the Model.models module
from Model.models import BiasModel
from utils.s3_manager import S3Manager


def main():
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

    # Create 'label' column based on 'Bias Score'
    df['label'] = df['Bias Score'].apply(lambda x: 'biased' if x > 0 else 'unbiased')

    # Convert DataFrame to list of examples
    examples = df.to_dict(orient='records')

    # Iterate over all OpenAI models
    for model_name in models:
        # Start a new W&B run for each model
        wandb.init(
            project="bias-detection-evaluation",
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

        # Initialize lists to collect true labels and predictions
        true_labels = []
        predicted_labels = []
        results = []

        for example in examples:
            label = example['label']
            question = example['Question']  # Adjust the key as per your data
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

        # Create a W&B Table for predictions
        predictions_table = wandb.Table(columns=["Question", "True Label", "Predicted Label", "Bias Score", "Bias Axes", "Debiased Text"])

        for example in results:
            predictions_table.add_data(
                example['Question'],
                example['label'],
                example['predicted_label'],
                example['bias_score'],
                ", ".join(example['bias_axes']) if example['bias_axes'] else None,
                example['debiased_text']
            )

        # Log evaluation metrics to W&B
        wandb.log({
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions_table
        })

        # Finish the W&B run
        wandb.finish()

        # Convert back to DataFrame for display
        results_df = pd.DataFrame(results)

        # Print the results DataFrame
        print(results_df[['Question', 'label', 'predicted_label']])

if __name__ == "__main__":
    main()
