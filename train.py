'''
T5 fine-tuning script with WandB logging modified from
https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
'''
import numpy as np
import pandas as pd
import torch
import os.path
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch_optimizer as optim
import wandb

# Check GPU availability
assert(cuda.is_available())
device = "cuda"


class C4200MDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[0]
        self.target_text = self.data[1]

    def __len__(self):
        return len(self.source_text)
    
    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Tokenize input and target sentences
        source_tokenized = self.tokenizer.encode_plus(source_text, truncation=True, max_length=self.source_len, padding=True, return_tensors="pt")
        target_tokenized = self.tokenizer.encode_plus(target_text, truncation=True, max_length=self.target_len, padding=True, return_tensors="pt")

        # Return tokenized sentences
        source_ids = source_tokenized["input_ids"].squeeze()
        source_mask = source_tokenized["attention_mask"].squeeze()
        target_ids = target_tokenized["input_ids"].squeeze()
        target_mask = target_tokenized["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }


def train(model, optimizer, loader, epoch=1):
    model.train()
    for i, data in enumerate(loader):
        src_ids = data["source_ids"].to(device, dtype=torch.long)
        src_mask = data["source_mask"].to(device, dtype=torch.long)
        tgt_ids = data["target_ids"].to(device, dtype=torch.long)
        # padded ids (pad=0) are set to -100, which means ignore for loss calculation
        tgt_ids[tgt_ids[: ,:] == 0 ] = -100
        label_ids = tgt_ids.to(device)

        # When we call model() with labels, they will be
        # - automatically right shifted by 1 (for teacher forcing)
        # - prepended by BOS=Beginning of sequence which is a PAD token
        # - any token that was -100 will be masked_fill_ to <pad> for teacher forcing
        outputs = model(input_ids=src_ids, attention_mask=src_mask, labels=label_ids)
        loss = outputs[0]

        # Log metrics with wandb and print
        if i%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        if i%500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, loader, tokenizer):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if i%100==0:
                print(f'Completed {i}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def main():
    # Init and config wandb
    wandb.init(project="GEC")
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 8     # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 8     # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 1         # number of epochs to train (default: 10)
    config.LEARNING_RATE = 1e-3     # learning rate (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3)
    config.SEED = 11411             # random seed (default: 42)
    config.MAX_LEN = 512

    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)     # numpy random seed
    torch.backends.cudnn.deterministic = True

    # T5 tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Import dataset
    DATASET_FOLDER = "/media/tailen/My Passport/c4200m"
    df = pd.read_csv(os.path.join(DATASET_FOLDER, "sentence_pairs.tsv-00000-of-00010"), sep='\t', header=None)

    # Split train and validation sets deterministically
    train_size = 0.9
    train_dataset = df.sample(frac=train_size, random_state = config.SEED)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    
    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    # Create the dataset and dataloader
    training_set = C4200MDataset(train_dataset, tokenizer, config.MAX_LEN, config.MAX_LEN)
    val_set = C4200MDataset(val_dataset, tokenizer, config.MAX_LEN, config.MAX_LEN)
    training_loader = DataLoader(training_set, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=8)

    # Load T5 model and optimizer
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)
    optimizer = optim.Adafactor(
        model.parameters(),
        lr=config.LEARNING_RATE,
        eps2=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False
    )

    # Log metrics with wandb
    wandb.watch(model, log="all")

    # Train and validate
    print("Preprocessing done. Starting training...")
    for epoch in range(config.TRAIN_EPOCHS):
        train(model, optimizer, training_loader, epoch)
        predictions, actuals = validate(model, val_loader, tokenizer)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv('./predictions.csv')
        print('Validation File generated for review')


if __name__ == "__main__":
    main()
