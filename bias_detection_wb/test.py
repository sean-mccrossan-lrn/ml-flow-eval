import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import pandas as pd

from utils.s3_manager import S3Manager

def load_data(file_path):
    return pd.read_csv(file_path)

def run_test():
    
    file_name = "/Users/seanmccrossan/Desktop/synthetic_data-2.csv"
    # Load the data
    df = load_data(file_name)

    # Load dataset from S3 using S3Manager
    bucket_name = 'learnosity-ds-datasets'
    object_name = 'data/bias/generated_synthetic_data.csv'

    s3_manager = S3Manager(bucket_name, region='us-east-1')

    # Upload the DataFrame to S3
    s3_manager.upload_dataframe(df, object_name=object_name)

    # Download the DataFrame from S3
    df_downloaded = s3_manager.download_file_to_dataframe(object_name=object_name)

    # Display the downloaded DataFrame
    print(df_downloaded)


if __name__ == "__main__":
    run_test()