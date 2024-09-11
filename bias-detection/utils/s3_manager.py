import boto3
import pandas as pd
import io
from botocore.exceptions import ClientError, NoCredentialsError

class S3Manager:
    def __init__(self, bucket_name, region=None):
        self.bucket_name = bucket_name
        self.region = region
        self.s3 = boto3.client('s3', region_name=region)

    def bucket_exists(self):
        """Check if the bucket exists and we have permission to access it."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} exists and you have permission to access it.")
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 403:
                print(f"Private Bucket. Forbidden Access! {e}")
            elif error_code == 404:
                print(f"Bucket {self.bucket_name} does not exist.")
            return False

    def create_bucket(self):
        """Create an S3 bucket in a specified region. If no region is specified, use the default region."""
        if not self.bucket_exists():
            try:
                if self.region is None or self.region == 'us-east-1':
                    self.s3.create_bucket(Bucket=self.bucket_name)
                else:
                    location = {'LocationConstraint': self.region}
                    self.s3.create_bucket(Bucket=self.bucket_name, CreateBucketConfiguration=location)
                print(f"Bucket {self.bucket_name} created successfully.")
            except ClientError as e:
                print(f"Failed to create bucket: {e}")

    def upload_file(self, file_name, object_name=None):
        """Uploads a file to the configured S3 bucket."""
        if object_name is None:
            object_name = file_name
        try:
            self.s3.upload_file(file_name, self.bucket_name, object_name)
            print(f"File {file_name} uploaded successfully to {object_name}.")
        except FileNotFoundError:
            print("The file was not found")
        except NoCredentialsError:
            print("Credentials not available")
        except ClientError as e:
            print(f"Failed to upload file: {e}")

    def download_file(self, object_name, file_name=None):
        """Downloads a file from the configured S3 bucket."""
        if file_name is None:
            file_name = object_name
        try:
            self.s3.download_file(self.bucket_name, object_name, file_name)
            print(f"File {file_name} downloaded successfully from {object_name}.")
        except FileNotFoundError:
            print("The file was not found")
        except NoCredentialsError:
            print("Credentials not available")
        except ClientError as e:
            print(f"Failed to download file: {e}")

    def upload_dataframe(self, df, object_name):
        """Upload a DataFrame to the S3 bucket as a CSV."""
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=object_name, Body=csv_buffer.getvalue())
            print(f"DataFrame uploaded successfully to {object_name}.")
        except ClientError as e:
            print(f"Failed to upload DataFrame: {e}")
            
    def download_file_to_dataframe(self, object_name):
        """Download a file from S3 and convert it to a DataFrame."""
        try:
            # Use a BytesIO buffer to hold the CSV data
            csv_buffer = io.BytesIO()

            # Check if the object exists first
            try:
                self.s3.head_object(Bucket=self.bucket_name, Key=object_name)
            except ClientError as e:
                print(f"Error: Could not find object {object_name} in bucket {self.bucket_name}.")
                return None

            # Download the object from S3
            self.s3.download_fileobj(self.bucket_name, object_name, csv_buffer)
            csv_buffer.seek(0)  # Move to the start of the buffer
            df = pd.read_csv(csv_buffer)
            return df
        except Exception as e:
            print(f"Failed to download or decode DataFrame: {e}")
            return None

    def list_objects(self):
        """List objects in the S3 bucket."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in response:
                for obj in response['Contents']:
                    print(f"Found object: {obj['Key']}")
            else:
                print("No objects found in the bucket.")
        except ClientError as e:
            print(f"Error listing objects: {e}")

    def list_buckets(self):
        """Lists all S3 buckets in your AWS account."""
        try:
            response = self.s3.list_buckets()
            for bucket in response['Buckets']:
                print(f"Bucket Name: {bucket['Name']} - Creation Date: {bucket['CreationDate']}")
        except ClientError as e:
            print(f"Failed to list buckets: {e}")
        except NoCredentialsError:
            print("Credentials not available")

    def delete_object_if_exists(self, object_name):
        """Check if an object exists in the S3 bucket and delete it if it does."""
        try:
            # Check if the object exists
            self.s3.head_object(Bucket=self.bucket_name, Key=object_name)
            # If the object exists, delete it
            self.s3.delete_object(Bucket=self.bucket_name, Key=object_name)
            print(f"Object '{object_name}' deleted successfully.")
        except self.s3.exceptions.NoSuchKey:
            # The error returned if the object does not exist
            print(f"No object named '{object_name}' was found in the bucket '{self.bucket_name}'.")
        except ClientError as e:
            # General AWS errors
            error_code = e.response['Error']['Code']
            if error_code == '403':
                print(f"Access Denied. Cannot access object '{object_name}'.")
            else:
                print(f"An error occurred: {e}")