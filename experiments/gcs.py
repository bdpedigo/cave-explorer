from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden


def test_storage_permissions():
    # Replace 'your-project-id' and 'your-bucket-name' with your actual project ID and bucket name
    project_id = "em-270621"
    bucket_name = "allen-minnie-phase3"
    

    # Create a storage client
    storage_client = storage.Client(project=project_id)

    try:
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Upload an object
        blob_name = "test-object.txt"
        blob = bucket.blob(blob_name)
        blob.upload_from_string("Hello, Cloud Storage!")
        print(f'Object "{blob_name}" uploaded successfully.')

        # Download the object
        downloaded_content = blob.download_as_text()
        print(f"Downloaded content: {downloaded_content}")

        # List objects in the bucket
        print("Objects in the bucket:")
        for blob in bucket.list_blobs():
            print(blob.name)

        # Delete the object
        blob.delete()
        print(f'Object "{blob_name}" deleted successfully.')

    except NotFound as e:
        print(f"Error: {e}. Bucket or object not found.")
    except Forbidden as e:
        print(f"Error: {e}. Permission denied.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_storage_permissions()
