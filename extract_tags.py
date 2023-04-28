import spacy
import errant
import argparse
import multiprocessing as mp

# Load the English spaCy model
nlp = spacy.load('en')
# Initialize ERRANT Annotator
annotator = errant.load('en', nlp)

def get_error_tags(orig_corr_tuple):
    orig, corr = orig_corr_tuple
    # Tokenize and parse the original and corrected sentences with spaCy
    original = annotator.parse(orig, tokenise=True)
    corrupted = annotator.parse(corr, tokenise=True)
    # Perform the ERRANT comparison
    edits = annotator.annotate(original, corrupted)
    # Extract the error tags
    error_tags = [(edit.type)[2:] for edit in edits]
    return error_tags

def process_input_file(input_file, output_file):
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        lines = [line.strip().split("\t") for line in fin]
        total_lines = len(lines)

        with mp.Pool() as pool:
            results = pool.imap_unordered(get_error_tags, lines)

            completed = 0
            for error_tags in results:
                for tag in error_tags:
                    fout.write(tag + "\t")
                fout.write("\n")
                completed += 1
                if completed % 5000 == 0:
                    print(f"Progress: {completed}/{total_lines} ({(completed / total_lines) * 100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, nargs='?', default=None)
    args = parser.parse_args()

    if args.input_file is None:
        while True:
            orig = input("Enter original sentence: ")
            corr = input("Enter corrected sentence: ")
            error_tags = get_error_tags((orig, corr))
            print("Error tags: " + str(error_tags))
    else:
        if not args.input_file.endswith(".tsv"):
            print("Input file must be a tsv file")
            exit(1)
        output_file = args.input_file[:-4] + "_tags.tsv"
        process_input_file(args.input_file, output_file)
