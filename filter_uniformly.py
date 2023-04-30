import random
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="File containing sentence pairs")
parser.add_argument("subset_size", type=int, help="Size of the subset to select")
args = parser.parse_args()

# Select subset uniformly at random
with open(args.data_path, "r") as f:
    original_lines = f.readlines()
subset_lines = random.sample(original_lines, args.subset_size)

# Write subset to file
output_path = args.data_path[:-4] + f"_{args.subset_size}.tsv"
with open(output_path, "w") as f:
    f.writelines(subset_lines)
