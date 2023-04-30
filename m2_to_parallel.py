# Convert a M2 file annotated by ERRANT to a parallel corpus of source and target sentences.
# Modified from https://github.com/asimokby/GEC-M2Format-to-Sentence
import argparse

def parse_annotation(annotation):
    splitted_annotation = annotation.split("|||")
    st_idx, end_idx = [int(i) for i in splitted_annotation[0].split()[1:]]
    operation_error = splitted_annotation[1].split(":") 
    if len(operation_error) > 2:
        operation, error = operation_error[0], ":".join(operation_error[1:])
    elif len(operation_error) == 2: 
        operation, error = operation_error
    else:
        if "UNK" in operation_error or "noop" in operation_error:
            operation = ""
            error = ""

    edit = splitted_annotation[2]
    if operation == "U": opOrder=1
    elif operation == "R": opOrder=2
    elif operation == "M": opOrder=3
    else: opOrder = 4

    return opOrder, st_idx, end_idx, operation, error, edit

def m2_to_correct(sent, annotations):
    tokens = sent.split()[1:]  # avoid the label "S"
    tokens_copy = tokens.copy()
    annotations_modifed = []
    for annotation in annotations:
        opOrder, st_idx, end_idx, operation, error, edit = parse_annotation(annotation)
        annotations_modifed.append((opOrder, st_idx,  end_idx, operation, error, edit))

    annotations_modifed = sorted(annotations_modifed, key=lambda x: x[0])
    for annot in annotations_modifed:
        start_index = annot[1]
        end_index = annot[2]
        diff = end_index - start_index 
        if annot[3] == "R":
            replacer = annot[5].strip().split()
            if diff - len(replacer) <= 0:
                div = len(replacer)//diff
                res = [replacer[i:i + div] for i in range(0, len(replacer), div)]
                replacer = [' '.join(i) for i in res]
            else: 
                replacer = replacer + ((diff-len(replacer))*[""])
            tokens_copy[start_index:end_index] = replacer
        elif annot[3] == "U": 
            tokens_copy[start_index:end_index] = [''] * diff
        elif annot[3] == "M":
            tokens_copy.insert(start_index, annot[5])
    edited_sentence = ' '.join(i for i in tokens_copy if i)
    original_sent = ' '.join(tokens)

    return original_sent, edited_sentence

def parse_m2_file_to_tuples(m2_file):
    with open(m2_file, 'r') as f:
        lines = f.readlines()

    sentence_annotation_tuples = []
    source = ""
    annotations = []

    for line in lines:
        line = line.strip()
        if line.startswith("S "):
            if source:
                sentence_annotation_tuples.append((source, annotations))
                source = ""
                annotations = []
            source = line
        elif line.startswith("A "):
            annotations.append(line)
    if source:
        sentence_annotation_tuples.append((source, annotations))

    return sentence_annotation_tuples

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("input_path", type=str, help="Path to the M2 file")
    args = argparse.parse_args()
    orig_path = args.input_path[:-3] + ".orig.txt"
    corr_path = args.input_path[:-3] + ".corr.txt"

    sentence_annotation_tuples = parse_m2_file_to_tuples(args.input_path)
    with open(orig_path, "w") as f_orig, open(corr_path, "w") as f_corr:
        for source, annotations in sentence_annotation_tuples:
            original_sent, edited_sent = m2_to_correct(source, annotations)
            f_corr.write(original_sent + "\n")
            f_orig.write(edited_sent + "\n")
