from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="mixedbread-ai/mxbai-embed-large-v1",
    local_dir="./models/mxbai-embed-large-v1",  # Or ~/.cache/huggingface/hub/models--mixedbread-ai--mxbai-embed-large-v1
    local_dir_use_symlinks=False
)
print(model_dir)  # Path to cached model
