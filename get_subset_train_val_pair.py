import pandas as pd
import numpy as np
import os.path
import argparse

DATASET_FOLDER = "/media/tailen/My Passport/c4200m"
np.random.seed(11411)

def get_subset_train_val_pair(SUBSET_PROPORTION=0.005, TRAIN_PROPORTION=0.9):
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
        print(f"Loaded {i+1}/10 shards, shape: {df.shape}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("subset_proportion", type=float, help="Proportion of the dataset to use")
    parser.add_argument("train_proportion", type=float, help="Proportion of the subset to use for training")
    args = parser.parse_args()
    get_subset_train_val_pair(args.subset_proportion, args.train_proportion)
