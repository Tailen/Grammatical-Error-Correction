'''
T5 fine-tuning script with WandB logging modified from
https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
'''
import numpy as np
import pandas as pd
import torch
import os.path
import wandb
import argparse
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Check GPU availability
assert(torch.cuda.is_available())
device = "cuda"


class C4200MDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.source_text = self.data[0]
        self.target_text = self.data[1]

    def __len__(self):
        return len(self.source_text)
    
    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Tokenize input and target sentences
        source_tokenized = self.tokenizer(source_text, truncation=True, max_length=self.max_len, return_tensors="pt")
        target_tokenized = self.tokenizer(target_text, truncation=True, max_length=self.max_len, return_tensors="pt")

        # Return input and target token ids, and attention mask
        return {
            "input_ids": source_tokenized["input_ids"][0],
            "attention_mask": source_tokenized["attention_mask"][0],
            "labels": target_tokenized["input_ids"][0],
        }


def main(train_file, val_file):
    # Init and config wandb
    wandb.init(project="GEC")
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 48    # batch size for training
    config.VALID_BATCH_SIZE = 256   # batch size for testing
    config.TRAIN_EPOCHS = 1         # number of epochs to train
    config.LEARNING_RATE = 1e-3     # learning rate (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3)
    config.SEED = 11411             # random seed (default: 42)
    config.MAX_LEN = 128            # max token length of input and output sequences

    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)     # numpy random seed

    # Load model, tokenizer, and data collator
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", return_tensors="pt")
    
    # Load datasets
    DATASET_FOLDER = "/media/tailen/My Passport/c4200m"
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, train_file), header=None)
    val_df = pd.read_csv(os.path.join(DATASET_FOLDER, val_file), header=None)
    train_dataset = C4200MDataset(train_df, tokenizer, config.MAX_LEN)
    val_dataset = C4200MDataset(val_df, tokenizer, config.MAX_LEN)

    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./weights",                                 # output directory
        evaluation_strategy="steps",                            # evaluation strategy to adopt during training
        prediction_loss_only=True,                              # only return the loss when evaluating
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,    # batch size per device during training
        per_device_eval_batch_size=config.VALID_BATCH_SIZE,     # batch size for evaluation
        learning_rate=config.LEARNING_RATE,                     # learning rate
        num_train_epochs=1,                                     # total number of training epochs
        eval_steps=6000,                                        # number of update steps between two evaluations
        save_steps=3000,                                        # after # steps model is saved
        save_total_limit=5,                                     # limit the total amount of checkpoints
        optim="adafactor",                                      # use adafactor optimizer to save VRAM
        tf32=True,                                              # enable 19-bit magic datatype
        bf16=True,                                              # enable mixed precision training
        load_best_model_at_end=True,                            # load best model at end of training
        # predict_with_generate=True,                             # use generate to calculate generative metrics (ROUGE, BLEU)
        report_to="wandb"                                       # report metrics to wandb
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, default="train.csv")
    parser.add_argument("val_file", type=str, default="val.csv")
    args = parser.parse_args()
    main(args.train_file, args.val_file)
