import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Parse arguments
parser = argparse.ArgumentParser()  
parser.add_argument("model_path", type=str)
args = parser.parse_args()

# Check GPU availability
assert(torch.cuda.is_available())
device = "cuda"

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=768)

# Run model inference loop
while True:
    input_sentence = input("Enter input sentence: ")
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    output_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Corrected sentence: " + output_sentence)
