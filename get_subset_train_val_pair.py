import pandas as pd
import numpy as np
import os.path

DATASET_FOLDER = "/media/tailen/My Passport/c4200m"
SUBSET_PROPORTION = 0.005 # 1M sentence pairs
TRAIN_PROPORTION = 0.9 # train/test split rate

np.random.seed(11411)

def get_subset_train_val_pair():
    # Get the subset of the dataset
    for i in range(10):
        file_name = os.path.join(DATASET_FOLDER, f"sentence_pairs.tsv-{i:05d}-of-00010")
        if i == 0:
            df = pd.read_csv(file_name, sep="\t", header=None)
            df = df.sample(frac=SUBSET_PROPORTION).reset_index(drop=True)
        else:
            df_new = pd.read_csv(file_name, sep="\t", header=None)
            df_new = df_new.sample(frac=SUBSET_PROPORTION).reset_index(drop=True)
            df = pd.concat([df, df_new], axis=0, ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    # Split the dataset into train and val sets
    train_size = int(len(df) * TRAIN_PROPORTION)
    df_train = df.iloc[:train_size, :]
    df_val = df.iloc[train_size:, :]
    print("Shape of train: {}, val: {}".format(df_train.shape, df_val.shape))

    # Save the train and val sets
    df_train.to_csv(os.path.join(DATASET_FOLDER, "train.csv"), header=None, index=None)
    df_val.to_csv(os.path.join(DATASET_FOLDER, "val.csv"), header=None, index=None)

if __name__ == "__main__":
    get_subset_train_val_pair()
