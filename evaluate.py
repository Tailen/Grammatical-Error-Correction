import numpy as np
import torch
import os.path
import argparse
import spacy
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration


'''
Output: (precision, recall, f1) on CoNLL14 test set using provided m2scorer
'''
def evalCoNLL14(model_path):
    pass

'''
Reference corrected sentences are hidden and model output needs to be submitted to
https://codalab.lisn.upsaclay.fr/competitions/4057#participate
'''
def evalBEA19(model_path):
    # Define paths
    dataset_path = "datasets_test/bea19-test-data/test.bea19.txt"
    output_path = "datasets_test/bea19-test-data/test.bea19.out"

    # Check GPU availability
    assert(torch.cuda.is_available())
    device = "cuda"

    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=768)

    # Load spaCy tokenizer
    nlp = spacy.load("en_core_web_sm")

    # Run model inference
    with open(dataset_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            input_ids = tokenizer.encode(line.strip(), return_tensors="pt").to(device)
            # Ignore very short and very long sentences
            if input_ids.shape[1] < 8 or input_ids.shape[1] > 128:
                output_sentence = line.strip()
            else:
                output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
                output_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Tokenize with spaCy
            doc = nlp(output_sentence)
            output_sentence = " ".join([token.text for token in doc])
            fout.write(output_sentence + "\n")

'''
Output: GLEU score on JFLEG test set using
https://www.nltk.org/api/nltk.translate.gleu_score.html
Dataset collected from https://huggingface.co/datasets/jfleg
'''
def evalJFLEG(model_path):
    eval_dataset = load_dataset("jfleg", split='test[:]')
    pass


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()  
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()
    
    # Run evaluations
    evalBEA19(args.model_path)
