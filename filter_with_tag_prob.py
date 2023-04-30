'''
Find a subset of the C4_200M dataset to approximate an error tag distribution
Usage: python filter_with_tag_prob.py <data_path> <subset_size> <iterations>

'''
import argparse
import random
from collections import defaultdict

def pretty_print_dist(d, target, decimals=4):
    total_error = 0
    for key, value in d.items():
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
    target_dist = {'ADJ': 0.012864718331629135, 
                   'ADJ:FORM': 0.0020093691294319075, 
                   'ADV': 0.01279337978265522, 
                   'CONJ': 0.004375431003733384, 
                   'CONTR': 0.0026157467957101752, 
                   'DET': 0.09465436473022139, 
                   'K': 0.02163935985542054, 
                   'MORPH': 0.015896606663020472, 
                   'NOUN': 0.03690580933583811, 
                   'NOUN:INFL': 0.00104629871828407, 
                   'NOUN:NUM': 0.03365990535752503, 
                   'NOUN:POSS': 0.005302832140394264, 
                   'ORTH': 0.04031816992842366, 
                   'OTHER': 0.10828002758423894, 
                   'PART': 0.00709818562290443, 
                   'PREP': 0.08290728366585023, 
                   'PRON': 0.022067391149264023, 
                   'PUNCT': 0.14751622951989157, 
                   'SPELL': 0.03294651986778589, 
                   'VERB': 0.049164150001188976, 
                   'VERB:FORM': 0.029771954438446723, 
                   'VERB:INFL': 0.0003804722611942073, 
                   'VERB:SVA': 0.01864314079851616, 
                   'VERB:TENSE': 0.05155399139181509, 
                   'WO': 0.01350676527239436, 
                   'SPACE': 0.0,
                   'noop': 0.15208189665422206
                   }
    
    # Add 18% fake noop to the end of existing list
    sample_count = len(tags)
    tags = tags + [["noop"]] * int(sample_count * 0.18)
    
    # Select the subset iteratively
    selected_tags = list(enumerate(tags))
    filter_size = (len(selected_tags) - args.subset_size) // args.iterations
    for i in range(args.iterations):
        print(f"\nIteration {i}")
        selected_tags = select_subset(selected_tags, target_dist, len(selected_tags) - filter_size)
    print("\nFinal distribution:")
    final_dist = get_distribution(selected_tags, target_dist)

    # Count and filter fake noop
    fake_noop_count = 1
    for i, tag_set in selected_tags:
        if i >= sample_count:
            fake_noop_count += 1
    selected_tags.sort(key=lambda x: x[0])
    selected_tags = selected_tags[:-fake_noop_count]
    selected_indices = [i for i, _ in selected_tags]

    # Choose indices of fake noop to add
    fake_noop_indices = []
    prev_index = -1
    for i in selected_indices:
        if len(fake_noop_indices) >= fake_noop_count:
            break
        if i - prev_index > 1:
            for j in range(prev_index+1, i):
                fake_noop_indices.append(j)
        prev_index = i
    print(f"Chose {len(fake_noop_indices)}/{fake_noop_count} fake noop indices")

    # Read the original sentence pair data
    with open(args.data_path, "r") as f:
        original_lines = f.readlines()
    original_lines = [line.strip().split("\t") for line in original_lines]
    for line in original_lines:
        if len(line) != 2:
            print("Incorrectly formatted line: " + line)
            exit(1)

    # Parse the subset as sentence pairs
    selected_indices_set = set(selected_indices)
    fake_noop_indices_set = set(fake_noop_indices)
    selected_lines = []
    noop_lines = []
    for i, line in enumerate(original_lines):
        if i in selected_indices_set:
            selected_lines.append(line)
        elif i in fake_noop_indices_set:
            noop_lines.append(line)
    noop_lines = [[line[1], line[1]] for line in noop_lines]
    selected_lines = selected_lines + noop_lines
    random.shuffle(selected_lines)

    # Save the selected subset
    output_path = args.data_path[:-4] + f"_{args.subset_size}.tsv"
    with open(output_path, "w") as f:
        f.writelines(["\t".join(line) + "\n" for line in selected_lines])
