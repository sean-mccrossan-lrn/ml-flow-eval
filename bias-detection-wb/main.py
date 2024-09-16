import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weave
from weave import Dataset
import json
from openai import OpenAI
from utils.s3_manager import S3Manager
import pandas as pd

@weave.op() # 🐝
def extract_fruit(sentence: str) -> dict:
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
            "role": "system",
            "content": "You will be provided with unstructured data, and your task is to parse it one JSON dictionary with fruit, color and flavor as keys."
        },
        {
            "role": "user",
            "content": sentence
        }
        ],
        temperature=0.7,
        response_format={ "type": "json_object" }
    )
    extracted = response.choices[0].message.content
    return json.loads(extracted)


class BiasModel(weave.Model):
    # Properties are entirely user-defined
    model_name: str
    system_message: str

    @weave.op()
    def predict(self, user_input):
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,  # Updated here
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_input},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


def main():
    weave.init('testing-1')  # 🐝
    bucket_name = 'learnosity-ds-datasets'
    object_name = 'data/bias/Synthetic_Complete_School_Homework_Questions_Dataset.csv'

    s3_manager = S3Manager(bucket_name, region='us-east-1')

    # Debug: List objects in the bucket
    print("Listing objects in the bucket...")
    s3_manager.list_objects()

    print(f"Attempting to download {object_name} from {bucket_name}...")
    df = s3_manager.download_file_to_dataframe(object_name)
    df['label'] = df['Bias Score'].apply(lambda x: 'biased' if x > 0 else 'unbiased')

    # Convert pandas DataFrame to a list of dictionaries for Weave Dataset
    data_list = df.to_dict(orient='records')

    # Create Dataset using the list of dictionaries
    dataset = Dataset(name='bias-detection-synthetic-test', rows=data_list)

    weave.publish(dataset)

    dataset_ref = weave.ref('bias-detection-synthetic-test:v0').get()

    # Access the 'rows' TableRef and convert it to a pandas DataFrame
    if hasattr(dataset_ref, 'rows'):
        table_ref = dataset_ref.rows
        print("TableRef found:", table_ref)

        # Convert rows to DataFrame
        rows_data = [dict(item) for item in table_ref]
        df_from_weave = pd.DataFrame(rows_data)

        # Print the DataFrame
        print("DataFrame from Weave:")
        print(df_from_weave.head())
    else:
        print("The dataset_ref object does not have a 'rows' attribute.")

    # Instantiate and use the BiasModel
    debiaser = BiasModel(
        model_name="gpt-3.5-turbo-1106",
        system_message="Return a structured JSON response containing a bias score (0 to 1), the specific axes of bias (e.g., gender, race, political), and a debiased version of the text, if that text is considered biased. Example of expected output: {\"bias_score\": 0.7, \"bias_axes\": [\"gender\", \"race\"], \"debiased_text\": \"Everyone should have equal opportunity.\"}",
    )

    result = debiaser.predict(
        "Mary is a great cook and mother, who likes to dress in nice clothes. John is strong and brave, and he likes to play sports. Who likes to play sports?"
    )
    print(result)


if __name__ == "__main__":
    main()