from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from transformers import BertModel, BertTokenizer
import torch
from sklearn.preprocessing import OneHotEncoder

model_name = 'nlpaueb/bert-base-greek-uncased-v1'
tokenizer = BertTokenizer.from_pretrained(model_name)

# the areas that will serve as target label indices
# idx2loc = {i:a for i,a in enumerate(train.area.unique())}
idx2loc = {0: 'Πόντος',
 1: 'Κύπρος',
 2: 'Κάρπαθος',
 3: 'Θεσπρωτία',
 4: 'Αμοργός',
 5: 'Σκύρος',
 6: 'Μικρά Ασία',
 7: 'Λέσβος',
 8: 'Μακεδονία',
 9: 'Λακωνία',
 10: 'Εύβοια',
 11: 'Επτάνησος',
 12: 'Αρκαδία',
 13: 'Νάξος',
 14: 'Κρήτη',
 15: 'Αχαΐα',
 16: 'Θράκη',
 17: 'Ιωάννινα',
 18: 'Αιτωλία',
 19: 'Κεφαλληνία',
 20: 'Ανατολική Θράκη',
 21: 'Ρόδος',
 22: 'Ήπειρος'}

loc2idx = {idx2loc[i]:i for i in idx2loc}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, max_length = 32):
        self.max_length = max_length
        self.labels = df.area.apply(lambda a: loc2idx[a])
        self.labels = np.array(self.labels.values)
        self.labels = np.reshape(self.labels, (self.labels.shape[0], 1))
        self.labels = OneHotEncoder(sparse_output=False).fit_transform(self.labels)
        self.texts = np.array(df.text.apply(lambda txt: tokenizer(txt, padding='max_length', max_length = self.max_length, truncation=True, return_tensors="pt")).values)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_labels = self.labels[idx]
        return batch_texts, batch_labels


class GrBertC(nn.Module):

    def __init__(self, dropout=0.1, num_classes=23):
        super(GrBertC, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 128, bias=True)
        self.norm = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, num_classes, bias=True)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = pooled_output
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.norm(x)
        x = self.linear2(x)
        return x
      

def validate(model, dataloader, device="cpu", criterion=nn.CrossEntropyLoss()):
    predictions, gold_labels = [], []
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_id, (val_input, val_label) in enumerate(dataloader):
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, val_label)
            gold = np.argmax(val_label.cpu().detach().numpy(), axis=1)
            pred = np.argmax(output.cpu().detach().numpy(), axis=1)
            predictions.extend(pred)
            gold_labels.extend(gold)
            val_loss += batch_loss.item()
    return predictions, gold_labels, val_loss/batch_id

  
def finetune(model, train_data, val_data, learning_rate=2e-5, epochs=10, criterion=nn.CrossEntropyLoss(), 
             batch_size=32, max_length=32, patience=2):
    
    train_losses = []
    val_losses = []
    
    train_dataloader = torch.utils.data.DataLoader(Dataset(train_data, max_length=max_length), 
                                                   batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(Dataset(val_data, max_length=max_length), 
                                                 batch_size=batch_size, drop_last=False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    lowest_loss = 10
    best_epoch = 0
    epochs_not_improving = 0
    for epoch_num in range(epochs):
            total_acc_train = 0
            total_loss_train = 0
            for batch_id, (inputs, labels) in tqdm(enumerate(train_dataloader)):
                model.train()
                output = model(inputs['input_ids'].squeeze(1).to(device), 
                               inputs['attention_mask'].to(device))
                batch_loss = criterion(output.to(device), labels.to(device))
                total_loss_train += batch_loss.item()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            train_losses.append(total_loss_train/(batch_id+1))
            
            predictions, gold_labels, val_loss = validate(model, val_dataloader, device, criterion)
            val_losses.append(val_loss)
            if val_loss < lowest_loss:
                print(f"New best epoch found: {epoch_num} (val loss: {val_loss:.3f})!")
                lowest_loss = val_loss
                best_epoch = epoch_num
                torch.save(model.state_dict(), "checkpoint.pt")
                epochs_not_improving = 0
            else:
                epochs_not_improving += 1
                if epochs_not_improving >= patience:
                    model.load_state_dict(torch.load("checkpoint.pt"))
                    print('Patience is up, restoring the best model and exiting...')
                    break
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/batch_id: .3f} \
                | Val Loss: {val_loss: .3f} (best epoch: {best_epoch} w/val_loss: {lowest_loss:.3f})')
    model.eval()    
    return model, train_losses, val_losses
