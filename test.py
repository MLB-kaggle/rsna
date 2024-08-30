import kagglehub

# Download latest version
path = kagglehub.model_download("anoukstein/simple_unet_2d_lspine/pyTorch/one")

print("Path to model files:", path)
