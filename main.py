from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
import torch.nn as nn
import sys
sys.path.append("/data0/lizhong/classify_bert_chinese")
if True:
    from classify_model import BertClassifier


labels = {
    'business': 0,
    'entertainment': 1,
    'sport': 2,
    'tech': 3,
    'politics': 4
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.labels = [labels[label] for label in df['Category']]
        self.texts = [
            tokenizer(text,
                      padding='max_length',
                      max_length=512,
                      truncation=True,
                      return_tensors="pt")
            for text in tqdm(df['Text'], desc="Tokenizing texts", total=len(df))
        ] # input_ids, token_type_ids, attention_mask
        pass

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def train():
    from torch.optim import Adam


def train(model, train_data, val_data, tokenizer, learning_rate, epochs):
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=16)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)  # [32, 1, 512]
            input_id = train_input['input_ids'].squeeze(1).to(device) # [32, 512]

            torch.cuda.empty_cache()  # TODO ???
            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # ------ 验证模型 -----------
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')


def evaluate(model, test_data, tokenizers):

    test = Dataset(test_data, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


if __name__ == "__main__":
    EPOCHS = 5
    LR = 1e-6
    np.random.seed(112)

    df = pd.read_csv("./BBC News Train.csv", delimiter=',')  # DEBUG df.head()

    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8*len(df)), int(.9*len(df))])

    print(len(df_train), len(df_val), len(df_test))

    model_dir = "/data0/lizhong/models/berts/bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertClassifier(model_dir=model_dir)
    train(model, df_train, df_val, tokenizer, LR, EPOCHS)

    evaluate(model, df_test, tokenizer)
    pass
