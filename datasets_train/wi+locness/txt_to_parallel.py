# Detokenize and combine two files (corrupted and original) into one parallel tsv file
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("corrupted_file", type=str, help="File containing corrupted sentences")
parser.add_argument("original_file", type=str, help="File containing original sentences")
parser.add_argument("output_file", type=str, help="Output file")
args = parser.parse_args()

def detokenize(line):
    # Remove spaces before punctuation
    line = re.sub(r'\s([?.!,";:](?:\s|$))', r'\1', line)
    # Join contractions
    line = re.sub(r"(\b\w+)(\s')(?=s|re|ve|ll|t|d|m|em)", r"\1'", line)
    # Handle "n't" contractions
    line = re.sub(r"(\b\w+)\s+n('t\b)", r"\1n\2", line)
    return line.strip()

with open(args.corrupted_file, "r") as f_corr, open(args.original_file, "r") as f_orig, open(args.output_file, "w") as f_out:
    for corr, orig in zip(f_corr, f_orig):
        f_out.write(detokenize(corr.strip()) + "\t" + detokenize(orig.strip()) + "\n")
