import pandas as pd

# Load the dataset, the loaded data is already in the form of tensors
data_root_dir = ""  # specify the correct data root directory
train_X = pd.read_pickle(f"{data_root_dir}/dataset/train_X.pkl")
train_Y = pd.read_pickle(f"{data_root_dir}/dataset/train_Y.pkl")
val_X = pd.read_pickle(f"{data_root_dir}/dataset/val_X.pkl")
val_Y = pd.read_pickle(f"{data_root_dir}/dataset/val_Y.pkl")
test_X = pd.read_pickle(f"{data_root_dir}/dataset/test_X.pkl")

print(
    "Data loaded successfully.",
    train_X.shape,
    train_Y.shape,
    val_X.shape,
    val_Y.shape,
    test_X.shape,
)
