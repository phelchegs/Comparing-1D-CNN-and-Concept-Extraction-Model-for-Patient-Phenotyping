import torch
from torch import nn
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score

class textCNN(nn.Module):
    def __init__(self, embeddings, vec_size, cnn_channels, cnn_windows, num_classes, dropout = 0.5):
        super().__init__()
        self.embed = nn.Embedding(embeddings.shape[0], vec_size)
        self.embed.weight.data.copy_(embeddings)
        self.convs = nn.ModuleList([nn.Conv2d(1, cnn_channels, (i, vec_size)) for i in cnn_windows])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(cnn_windows)*cnn_channels, num_classes)
        
    def forward(self, input):
        x = self.embed(input)
        x = x.unsqueeze(1) #add the channel dimension
        conv_x = [layer(x).squeeze(-1) for layer in self.convs]
        max_x = [F.max_pool1d(nn.ReLU()(x), x.size()[-1]) for x in conv_x]
        concat_x = torch.concat(max_x, dim = 1).squeeze(-1) #concat along the channel dimension
        dropout_x = self.dropout(concat_x)
        logit_x = self.fc(dropout_x)
        return logit_x
        
def train_one_epoch(model, dataloader, device, optimizer, weights, train_state):
    start_time = time.time()
    model.train()
    corrects = 0
    one_epoch_loss = 0
    one_epoch_tokens = 0
    one_epoch_samples = 0
    batch_loss = 0
    batch_tokens = 0
    for i, b in enumerate(dataloader):
        x, y = b.x.to(device), b.y.to(device)
        one_epoch_samples += x.shape[0]
        logit = model(x)
        loss_node = nn.CrossEntropyLoss(weight = weights)(logit, y)
        loss_node.backward()
        optimizer.zero_grad(set_to_none = True)
        optimizer.step()
        # scheduler.step()
        result = torch.max(logit, dim = 1)[1]
        corrects += (result.view(b.y.size()).data == b.y.data).sum()
        train_state.step += 1
        train_state.samples += b.x.shape[0]
        train_state.tokens += b.tokens
        one_epoch_loss += loss_node.data
        one_epoch_tokens += b.tokens
        batch_loss += loss_node.data
        batch_tokens += b.tokens
        if i%10 == 1:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print('Batch {} in the current training epoch, loss per batch {:.2f}, tokens per second {:.2f}, accuracy {:.2%}.'.format(i, batch_loss, batch_tokens/elapsed, corrects/one_epoch_samples))
            batch_loss = 0
            batch_tokens = 0
            start_time = time.time()
        del loss_node
    return one_epoch_loss, one_epoch_tokens, train_state
    
def val_one_epoch(model, dataloader, weights, device):
    model.eval()
    val_loss = 0
    corrects = 0
    data_num = 0
    y_true = []
    y_pred = []
    for i, b in enumerate(dataloader):
        x, y = b.x.to(device), b.y.to(device)
        y_true += y.tolist()
        logit = model(x)
        loss_node = nn.CrossEntropyLoss(weight = weights)(logit, y)
        val_loss += loss_node.data
        result = torch.max(logit, dim = 1)[1]
        corrects += (result.view(b.y.size()).data == b.y.data).sum()
        data_num += x.shape[0]
        y_pred += result.tolist()
        if i%10 == 1:
            print('Batch {} of the current validating epoch, loss per obs {:.2f}, accuracy {:.2%}.'.format(i, val_loss/data_num, corrects/data_num))
    auc_score = roc_auc_score(y_true, y_pred)
    print('Auc score of the validation in the current epoch {}'.format(auc_score))
    return val_loss, corrects/data_num