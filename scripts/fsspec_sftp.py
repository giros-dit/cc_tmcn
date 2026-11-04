import fsspec
import sftp_config as sftp_config
import argparse

fs = fsspec.filesystem(
    "simplecache",
    target_protocol="sftp",
    target_options=sftp_config.sftp_options,
    cache_storage="/home/jovyan/disco6g-data-cache",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SFTP connection and list files in remote directory")
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to dataset directory on the SFTP server')
    args = parser.parse_args()
    
    # Test the SFTP connection and list files in the remote directory
    remote_dir = args.dataset_path  # Use the dataset path provided as an argument
    try:
        files = fs.ls(remote_dir)
        print(f"Files in {remote_dir}:")
        for file in files:
            print(file)
    except Exception as e:
        print(f"Error accessing {remote_dir}: {e}")