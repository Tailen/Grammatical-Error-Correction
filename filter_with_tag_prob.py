'''
Find a subset of the C4_200M dataset to approximate an error tag distribution
Usage: python filter_with_tag_prob.py <data_path> <subset_size> <iterations>

'''
import argparse
import random
from collections import defaultdict

def pretty_print_dist(d, target, decimals=4):
    total_error = 0
    for key, value in sorted(d.items()):
        total_error += abs(target[key]-value)
        print(f"{key}:\t{value:.{decimals}f}\t{(target[key]-value):.{decimals}f}")
    print(f"Total error: {total_error:.{decimals}f}")

def get_distribution(tags, target, decimals=4):
    dist = defaultdict(int)
    for _, tag_set in tags:
        for tag in tag_set:
            dist[tag] += 1
    total_tags = sum(dist.values())
    dist = {tag: count / total_tags for tag, count in dist.items()}
    pretty_print_dist(dist, target, decimals)
    return dist

def select_subset(tags, target_dist, subset_size):
    # Calculate the current distribution of error tags in the dataset
    current_dist = get_distribution(tags, target_dist)

    # Calculate the score for each tag set based on the current and target distributions
    tag_scores = []
    for i, tag_set in tags:
        score = 0
        for tag in tag_set:
            score += target_dist[tag] - current_dist[tag]
        score /= len(tag_set)
        tag_scores.append((i, tag_set, score))

    # Sort the tag sets by their scores in descending order
    tag_scores.sort(key=lambda x: x[2], reverse=True)
    # Select the top subset_size tag sets
    selected_tags = [(i, tag_set) for i, tag_set, _ in tag_scores[:subset_size]]

    return selected_tags

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="File containing sentence pairs")
    parser.add_argument("subset_size", type=int, help="Size of the subset to select")
    parser.add_argument("iterations", type=int, help="Number of iterations to run")
    args = parser.parse_args()

    # Parse tag file
    tag_path = args.data_path[:-4] + "_tags.tsv"
    with open(tag_path, "r") as f:
        lines = f.readlines()
    tags = [line.strip().split("\t") for line in lines]
    tags = [["noop"] if len(tag_set)==1 and tag_set[0]=="" else tag_set for tag_set in tags]

    # Define the target distribution of error tags
    target_dist = {'ADJ': 0.014653665623936019, 
                   'ADJ:FORM': 0.0022887888081748495, 
                   'ADV': 0.01457240684968129, 
                   'CONJ': 0.004983871487623341, 
                   'CONTR': 0.002979488389340041, 
                   'DET': 0.10781685030698211, 
                   'K': 0.024648494857267608, 
                   'MORPH': 0.018107163529761973, 
                   'NOUN': 0.042037872547779485, 
                   'NOUN:INFL': 0.0011917953557360162, 
                   'NOUN:NUM': 0.03834059831918934, 
                   'NOUN:POSS': 0.00604023555293481, 
                   'ORTH': 0.045924750582964, 
                   'OTHER': 0.12333727618963523, 
                   'PART': 0.008085248038345474, 
                   'PREP': 0.09443623881303684, 
                   'PRON': 0.02513604750279598, 
                   'PUNCT': 0.16802960202973585, 
                   'SPELL': 0.03752801057664206, 
                   'VERB': 0.056000838590550314, 
                   'VERB:FORM': 0.033911995122306644, 
                   'VERB:INFL': 0.00043338012935855135, 
                   'VERB:SVA': 0.021235626338569017, 
                   'VERB:TENSE': 0.05872300752808371, 
                   'WO': 0.015384994592228574, 
                   'SPACE': 0.0, 
                   'noop': 0.034171752337340926
                   }
    
    # Select the subset iteratively
    selected_tags = list(enumerate(tags))
    filter_size = (len(selected_tags) - args.subset_size) // args.iterations
    for i in range(args.iterations):
        print(f"\nIteration {i}")
        selected_tags = select_subset(selected_tags, target_dist, len(selected_tags) - filter_size)
    print("\nFinal distribution:")
    final_dist = get_distribution(selected_tags, target_dist)

    # Read the original sentence pair data
    with open(args.data_path, "r") as f:
        original_lines = f.readlines()
    original_lines = [line.strip().split("\t") for line in original_lines]
    for line in original_lines:
        if len(line) != 2:
            print("Incorrectly formatted line: " + line)
            exit(1)

    # Parse the subset as sentence pairs
    selected_tags.sort(key=lambda x: x[0])
    selected_indices = [i for i, _ in selected_tags]
    selected_indices_set = set(selected_indices)
    selected_lines = []
    for i, line in enumerate(original_lines):
        if i in selected_indices_set:
            selected_lines.append(line)
    random.shuffle(selected_lines)

    # Save the selected subset
    output_path = args.data_path[:-4] + f"_{args.subset_size}.tsv"
    with open(output_path, "w") as f:
        f.writelines(["\t".join(line) + "\n" for line in selected_lines])
