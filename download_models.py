import os
import subprocess
from huggingface_hub import snapshot_download

def create_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Directory created: {path}')
    else:
        print(f'Directory already exists: {path}')

def download_file(url, folder_path, file_name=None):
    """Download a file from a given URL to a specified folder with an optional file name using aria2c."""
    local_filename = file_name if file_name else url.split('/')[-1]
    local_filepath = os.path.join(folder_path, local_filename)

    # Check if file exists
    if os.path.exists(local_filepath):
        print(f'File already exists: {local_filepath}')
        return

    print(f'Downloading {url} to: {local_filepath}')
    
    # Download using aria2c
    aria2c_command = [
        'aria2c',
        '-c',  # Continue downloading if the file is partially downloaded
        '-x', '16',  # Number of connections per server
        '-s', '16',  # Number of download parts
        '-k', '1M',  # Size of a segment
        '-d', folder_path,  # Directory to save the file
        '-o', local_filename,  # Output file name
        url
    ]
    
    # Run the aria2c command
    result = subprocess.run(aria2c_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f'Error downloading file: {result.stderr.decode()}')
    else:
        print(f'Downloaded {local_filename} to {folder_path}')

if __name__ == '__main__':
    # Define the folders and their corresponding file URLs with optional file names
    folders_and_files = {
        os.path.join('models'): [
            ('https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin', None),
            ('https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0Q_fp16.safetensors', 'v0Q.safetensors')
        ]
    }

    for folder, files in folders_and_files.items():
        create_directory(folder)
        for file_url, file_name in files:
            download_file(file_url, folder, file_name)

    # Uncommented the model download parts to use aria2c instead
    # llava_model = os.getenv('LLAVA_MODEL', 'liuhaotian/llava-v1.5-7b')
    # print(f'Downloading LLaVA model: {llava_model}')
    # model_folder = llava_model.split('/')[1]
    # snapshot_download(llava_model, local_dir=os.path.join("models", model_folder), local_dir_use_symlinks=False)

    # llava_clip_model = 'openai/clip-vit-large-patch14-336'
    # print(f'Downloading LLaVA CLIP model: {llava_clip_model}')
    # model_folder = llava_clip_model.split('/')[1]
    # snapshot_download(llava_clip_model, local_dir=os.path.join("models", model_folder), local_dir_use_symlinks=False)

    sdxl_clip_model = 'openai/clip-vit-large-patch14'
    print(f'Downloading SDXL CLIP model: {sdxl_clip_model}')
    model_folder = sdxl_clip_model.split('/')[1]
    snapshot_download(sdxl_clip_model, local_dir=os.path.join("models", model_folder), local_dir_use_symlinks=False)
