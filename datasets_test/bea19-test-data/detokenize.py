import re

def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def detokenize(line):
    # Remove spaces before punctuation
    line = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', line)

    # Join contractions
    line = re.sub(r"(\b\w+)(\s')(?=s|re|ve|ll|t|d|m|em)", r"\1'", line)

    return line.strip()

if __name__ == "__main__":
    input_file = "ABCN.test.bea19.orig"
    output_file = "test.bea19.txt"

    lines = read_lines(input_file)
    detokenized_lines = [detokenize(line) for line in lines]

    with open(output_file, 'w') as file:
        for line in detokenized_lines:
            file.write(line + '\n')
            