import kagglehub

# Download latest version
path = kagglehub.dataset_download("bogdancretu/flower299")

print("Path to dataset files:", path)