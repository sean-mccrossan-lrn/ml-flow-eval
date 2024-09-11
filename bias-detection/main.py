import pandas as pd
from utils.s3_manager import S3Manager
import json
import numpy as np
import io
import mlflow
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay, f1_score
# import dagshub
import os
from dotenv import load_dotenv
import dagshub
from langsmith import Client
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('punkt')
from nltk.util import ngrams
from rouge import Rouge
import json



# Load environment variables from .env file
load_dotenv()


def proportional_sample(data, total_samples):
    """Sample approximately proportionally from each group."""
    counts = data['label'].value_counts(normalize=True)
    samples_per_group = (counts * total_samples).round().astype(int)
    return data.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), samples_per_group[x.name]), random_state=42)
    )

def download_dataframe_and_print(bucket_name, object_name, region='us-east-1'):
    """Download a CSV file from S3 and print it as a DataFrame."""
    s3_manager = S3Manager(bucket_name, region)
    df_content = s3_manager.download_file_to_dataframe(object_name)
    if df_content is not None and not df_content.empty:
        print("Downloaded DataFrame:")
        print(df_content.head())
    else:
        print("Failed to download or parse DataFrame.")

def calculate_jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_metrics(df):
    # Initialize metrics columns with NaNs
    df['Cosine Similarity'] = np.nan
    df['BLEU Score'] = np.nan
    df['Jaccard Similarity'] = np.nan
    df['ROUGE-1'] = np.nan
    df['ROUGE-2'] = np.nan
    df['ROUGE-L'] = np.nan

    # Filter the DataFrame to only process rows where both label_encoded and predicted_bias are True
    filtered_df = df[(df['label_encoded'] == True) & (df['predicted_bias'] == True)]

    # Compute metrics only if there are any rows to process
    if not filtered_df.empty:
        vectorizer = TfidfVectorizer()
        texts = filtered_df['Debiased Question'].tolist() + filtered_df['debiased_text'].tolist()
        vecs = vectorizer.fit_transform(texts)
        cos_sim = cosine_similarity(vecs[:len(filtered_df)], vecs[len(filtered_df):])
        df.loc[filtered_df.index, 'Cosine Similarity'] = np.diag(cos_sim)

        # Compute BLEU, Jaccard, and ROUGE
        rouge = Rouge()
        for index, row in filtered_df.iterrows():
            reference = row['Debiased Question']
            candidate = row['debiased_text']

            # BLEU
            bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=SmoothingFunction().method1)
            df.at[index, 'BLEU Score'] = bleu_score

            # Jaccard
            jaccard_score = calculate_jaccard_similarity(reference, candidate)
            df.at[index, 'Jaccard Similarity'] = jaccard_score

            # ROUGE
            scores = rouge.get_scores(candidate, reference)[0]
            df.at[index, 'ROUGE-1'] = scores['rouge-1']['f']
            df.at[index, 'ROUGE-2'] = scores['rouge-2']['f']
            df.at[index, 'ROUGE-L'] = scores['rouge-l']['f']

    return df


def run_experiment(df, client):
    mlflow.set_experiment("test-gpt-4o-bias-evaluation-proxy")
    dataset_source_url = "s3://learnosity-ds-datasets/data/bias/Synthetic_Complete_School_Homework_Questions_Dataset.csv"
    # Create an instance of a PandasDataset
    dataset = mlflow.data.from_pandas(
        df, source=dataset_source_url, name="Assesment Validation Bias"
    )

    with mlflow.start_run():
        model_used = "gpt-4-turbo"
        prompt = "Return a structured JSON response containing a bias score (0 to 1), the specific axes of bias (e.g., gender, race, political), and a debiased version of the text, if that text is considered biased. Example of expected output: {\"bias_score\": 0.7, \"bias_axes\": [\"gender\", \"race\"], \"debiased_text\": \"Everyone should have equal opportunity.\"}"

        mlflow.log_param("model", model_used)
        mlflow.log_param("prompt", prompt)
        mlflow.log_input(dataset=dataset, context="validation")

        bias_results = []

        for index, row in df.iterrows():
            comment = row['Question']
            response = client.chat.completions.create(
                model=model_used,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Evaluate the following text for bias: '{comment}'"}
                ]
            )

            try:
                content = response.choices[0].message.content  # Access content directly
                # Parse the content as JSON
                result = json.loads(content)
                bias_score = result['bias_score']
                bias_axes = result['bias_axes']
                debiased_text = result['debiased_text']

                bias_results.append({
                    'bias_score': bias_score,
                    'bias_axes': bias_axes,
                    'debiased_text': debiased_text
                })
            except json.JSONDecodeError as e:
                print("JSON Decode Error:", e)
                continue
            except KeyError as e:
                print("Key error in JSON response:", e)
                continue

        results_df = pd.DataFrame(bias_results)
        df = pd.concat([df, results_df], axis=1)

        df['label_encoded'] = df['label'].map({'biased': True, 'unbiased': False})
        df['predicted_bias'] = df['bias_score'] > 0.5
        cm = confusion_matrix(df['label_encoded'], df['predicted_bias'], labels=[True, False])

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        df = calculate_metrics(df)  # Calculate similarity metrics

        # Calculate and log other metrics
        accuracy = accuracy_score(df['label_encoded'], df['predicted_bias'])
        f1_score_result = f1_score(df['label_encoded'], df['predicted_bias'])
        recall = recall_score(df['label_encoded'], df['predicted_bias'], pos_label=True)
        specificity = recall_score(df['label_encoded'], df['predicted_bias'], pos_label=False)

        # Log the new metrics
        mlflow.log_metric("BLEU Score", df['BLEU Score'].mean())
        mlflow.log_metric("Jaccard Similarity", df['Jaccard Similarity'].mean())
        mlflow.log_metric("Cosine Similarity", df['Cosine Similarity'].mean())
        mlflow.log_metric("ROUGE-1", df['ROUGE-1'].mean())
        mlflow.log_metric("ROUGE-2", df['ROUGE-2'].mean())
        mlflow.log_metric("ROUGE-L", df['ROUGE-L'].mean())
    
        # Log evaluation dataset
        eval_path = "eval_dataset.csv"
        df.to_csv(eval_path, index=False)
        mlflow.log_artifact(eval_path)

        # Export to JSON file
        json_file_path = "eval_dataset.json"
        df.to_json(json_file_path, orient='split', index=False)

        mlflow.log_table(data=df, artifact_file=json_file_path)

        # Log old metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score_result)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("specificity", specificity)

        mlflow.end_run()
        
        return df

def main():
    bucket_name = 'learnosity-ds-datasets'
    object_name = 'data/bias/Synthetic_Complete_School_Homework_Questions_Dataset.csv'
    dagshub.init(repo_owner='sean-mccrossan-lrn', repo_name='ml-flow-eval', mlflow=True)
    # file_name = '/Users/seanmccrossan/Downloads/Complete_School_Homework_Questions_Dataset.csv'
    # object_name = 'data/bias_proxy_sample.csv'
    # upload_object_name = 'data/bias/Synthetic_Complete_School_Homework_Questions_Dataset.csv'

    # Initialize S3Manager
    s3_manager = S3Manager(bucket_name, region='us-east-1')

    # Debug: List objects in the bucket
    print("Listing objects in the bucket...")
    s3_manager.list_objects()
    # client=Client()

    # print(client.list_datasets())

    # print(f"Uploading DataFrame to {bucket_name} as {upload_object_name}...")
    # s3_manager.upload_file(file_name, upload_object_name)

    print(f"Attempting to download {object_name} from {bucket_name}...")
    df = s3_manager.download_file_to_dataframe(object_name)
    df['label'] = df['Bias Score'].apply(lambda x: 'biased' if x > 0 else 'unbiased')

    
    if df is not None:
        print("Downloaded DataFrame:")
        print(df.head())
    else:
        print("Failed to download or parse DataFrame.")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    eval_df = run_experiment(df, client)
    print(eval_df.head())


if __name__ == '__main__':
    main()
