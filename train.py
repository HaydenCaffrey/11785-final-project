# %%
config = {
    "batch_size":8,
    "beam_width" : 2,
    "lr" : 1e-5,
    "weight_decay": 0,
    "epochs" : 100
    } # Feel free to add more items here


# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
# import torch
import torch
import torch.nn as nn
import numpy as np
from torchsummaryX import summary
import sklearn
import gc
import zipfile
import pandas as pd
from tqdm.auto import tqdm
import os
import datetime
import wandb
from torch.nn.utils.rnn import pad_sequence
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
torch.manual_seed(0)
# model_name = "facebook/wav2vec2-large-xlsr-53"
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# model_audio = Wav2Vec2Model.from_pretrained(model_name)

# i= feature_extractor(data, return_tensors="pt", sampling_rate=16000)
# #previous are in dataloader

# with torch.no_grad():
#   o= model_audio(i.input_values)
# print(o.keys())
# print(o.last_hidden_state.shape)
# print(o.extract_features.shape)
# print(i.input_values.numpy().shape)

# %%
from sentence_transformers import SentenceTransformer
model_text = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Sentences we want to encode. Example:
# sentence = ['This framework generates embeddings for each input sentence.']


#Sentences are encoded by calling model.encode()
# text_fea = model_text.encode(sentence)

# %%
import json

class RASDataset(torch.utils.data.Dataset):

    def __init__(self, root, file_pth, partition = 'train', subset= None): 
        # Load the directory and all files in them
        f = open(file_pth)
        self.data_json = json.load(f)
        f.close()

        self.length = len(self.data_json)       
        self.base_path = os.path.join(root, partition)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cur_dict = self.data_json[idx]
        name = str(cur_dict['id'])

        audio_feat = np.load(os.path.join(self.base_path, name+'_feat.npy'))
        # audio_feat = np.load(os.path.join(self.base_path, name+'_hidden_state.npy'))
        # print('**************',audio_feat.shape)
        
        audio_feat = np.squeeze(audio_feat)

        text = [cur_dict['description']]
        text_feat = model_text.encode(text)
        text_feat = np.squeeze(text_feat)

        target = np.load(os.path.join(self.base_path, name+'_target.npy'))
        target = np.squeeze(target)
        
        audio_feat = torch.FloatTensor(audio_feat)
        text_feat = torch.FloatTensor(text_feat)
        target = torch.LongTensor(target)

        sample = {
                  "audio_feat": audio_feat,
                  "text_feat": text_feat,
                  "target": target
                }

        return sample


    def collate_fn(self,batch):

        batch_audio = [i["audio_feat"] for i in batch]
        batch_text = [i["text_feat"] for i in batch]
        batch_target = [i["target"] for i in batch]

        batch_audio_pad = pad_sequence(batch_audio, batch_first=True)
        lengths_audio = [i.shape[0] for i in batch_audio]

        batch_target_pad = pad_sequence(batch_target, batch_first=True)
        lengths_target = [i.shape[0] for i in batch_target]

        batch_audio_pad = torch.FloatTensor(batch_audio_pad)
        batch_text = torch.stack(batch_text)
        batch_target_pad = torch.LongTensor(batch_target_pad)

        return batch_audio_pad, batch_text, batch_target_pad, torch.tensor(lengths_audio), torch.tensor(lengths_target)

       

# %%
root = './'
train_data = RASDataset(root, 'train_combined.json', partition= "train")
val_data = RASDataset(root, 'val_combined.json', partition= "val")
test_data = RASDataset(root, 'test_combined.json', partition= "test")

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data, 
    num_workers = 0,
    batch_size  = config['batch_size'], 
    collate_fn = train_data.collate_fn,
    pin_memory  = True,
    shuffle     = True
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data, 
    num_workers = 0,
    batch_size  = config['batch_size'],
    collate_fn = val_data.collate_fn,
    pin_memory  = True,
    shuffle     = False
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data, 
    num_workers = 0,
    batch_size  = config['batch_size'],
    collate_fn = test_data.collate_fn,
    pin_memory  = True,
    shuffle     = False
)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))

for data in train_loader:
    audio, text, target, laudio, ltarget = data
    print(audio.shape, text.shape, target.shape, laudio.shape, ltarget.shape)
    break 

# class Attention(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.Tanh(),
#         )
#         self.linear = nn.Linear(hidden_size, 1, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         outputs = self.linear(self.fc(x))
#         # print(outputs.size())
#         alpha = torch.softmax(outputs, dim=2)
#         x = (x * alpha)
#         return x

# class RASModel(torch.nn.Module):

#     def __init__(self, embed_dim, num_heads):
#         super().__init__()

#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.audio_linear = nn.Linear(512, self.embed_dim)
#         self.text_linear = nn.Linear(384, 384)
#         # self.attention = nn.MultiheadAttention(self.embed_dim+384, self.num_heads, batch_first=True)
#         self.attention = Attention(self.embed_dim+384,self.embed_dim+384) #nn.MultiheadAttention(self.embed_dim+384, self.num_heads, batch_first=True)
#         self.fc1 = nn.Linear(self.embed_dim+384, 2)
#         self.fc2 = nn.Linear(512,124)
#         self.fc3 = nn.Linear(124,2)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.rnn = nn.LSTM(self.embed_dim+384,self.embed_dim+384,2,batch_first=True)

        
    
#     def forward(self, audio_fea, text_fea):

#         # audio_fea = self.audio_linear(audio_fea)
#         # text_fea = self.text_linear(text_fea)

#         B, T, D = audio_fea.size()

#         text_fea_rep = text_fea.repeat(T, 1) #B,512 -> T,B,512
#         text_fea_rep = text_fea_rep.reshape(B,T,-1)

#         x = torch.cat((audio_fea, text_fea_rep), 2)

#         # x, _ = self.attention(x, x, x)
#         # x, _ = self.rnn(x)
#         # print(rnn_output.size())
        
#         x = self.attention(x)
#         # x, _ = self.rnn(x)

#         # print(att_output.size())
#         linear_attn = self.fc1(x)
#         # print(linear_attn.size())
#         return linear_attn

class Attention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(self.fc(x))
        # print(outputs.size())
        alpha = torch.softmax(outputs, dim=2)
        x = (x * alpha)
        return x

class RASModel(torch.nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.audio_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_linear = nn.Linear(384, self.embed_dim)
        self.mha_a_t = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                               dropout=self.dropout, batch_first=True)
        self.mha_t_a = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                               dropout=self.dropout, batch_first=True)
        self.fc1 = nn.Linear(self.embed_dim*2, 2)
        self.fc2 = nn.Linear(self.embed_dim,124)
        self.fc3 = nn.Linear(124,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.rnn = nn.LSTM(self.embed_dim,self.embed_dim,2,batch_first=True)
        self.concat_linear = nn.Linear(in_features=2 * self.embed_dim, out_features= self.embed_dim)
        self.classifier = nn.Linear(in_features= self.embed_dim, out_features=2)
        
    
    def forward(self, audio_fea, text_fea):

        B, T, D = audio_fea.size()
        text_fea = text_fea[:,None,:]
        text_fea_rep = text_fea.repeat(1, T, 1) #B,1, 384 -> B, T, 384

        audio_fea = self.audio_linear(audio_fea)
        text_fea = self.text_linear(text_fea_rep)

        x_a2t, _ = self.mha_a_t(text_fea, audio_fea, audio_fea)
        # x_a2t = torch.mean(x_a2t, dim=2)

        x_t2a, _ = self.mha_t_a(audio_fea, text_fea, text_fea)
        # x_t2a = torch.mean(x_t2a, dim=2)

        x = torch.stack((x_a2t, x_t2a), dim=2)
        x_mean, x_std = torch.std_mean(x, dim=2)
        x = torch.cat((x_mean, x_std), dim=2)  
        x = self.concat_linear(x)
        # x,_ = self.rnn(x)
        x = self.classifier(x)
        return x

model = RASModel(embed_dim = 512, num_heads = 8, dropout=0.2).to(device)
# model = RASModel(embed_dim = 512, num_heads = 8).to(device)
summary(model, audio.to(device), text.to(device))

class weighted_log_loss(nn.Module):    
    def __init__(self):
        super(weighted_log_loss,self).__init__()
        self.LOSS_BIAS = 0.2

    def forward(self, yt, yp):   
        pos_loss = -(0 + yt) * torch.log(0 + yp + 1e-7)
        neg_loss = -(1 - yt) * torch.log(1 - yp + 1e-7)

        return self.LOSS_BIAS * torch.mean(neg_loss) + (1. - self.LOSS_BIAS) * torch.mean(pos_loss)


import numpy as np
criterion = nn.CrossEntropyLoss((torch.FloatTensor([0.3, 1]).to(device)))
# criterion = nn.L1Loss()

optimizer =  torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']) # What goes in here?
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)


from sklearn.metrics import f1_score

def train(model, dataloader, optimizer, criterion):

    model.train()
    tloss, tacc = 0, 0 # Monitoring loss and accuracy
    
    prob_all = []
    label_all = []

    batch_bar   = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    
    for i, (audio_fea, text_fea, target, audio_len, target_len) in enumerate(dataloader):
        
        ### Initialize Gradients
        optimizer.zero_grad()

        ### Move Data to Device (Ideally GPU)
        audio_fea = audio_fea.to(device)
        text_fea  = text_fea.to(device)
        target = target.to(device)

        ### Forward Propagation
        logits  = model(audio_fea, text_fea).permute(0,2,1)

        ### Loss Calculation
        # for j in range(logits.shape[2]):
        #     if j==0:
        #         loss  = criterion(logits[:,:,j], target[:,j])
        #     else:
        #         loss  += criterion(logits[:,:,j], target[:,j])
        loss = criterion(logits,target)
        ### Backward Propagation
        loss.backward() 
        
        ### Gradient Descent
        optimizer.step()       

        tloss   += loss.item()
        for j in range(logits.shape[0]):
            # print(torch.argmax(logits[j,:,:audio_len[j]], dim= 0).shape)
            # print(target[j,:target_len[j]].shape)
            tacc += torch.mean(torch.argmax(logits[j,:,:audio_len[j]], dim= 0) == target[j,:target_len[j]], dtype=torch.float32).item()
            prob_all.extend(np.argmax(logits.detach().cpu().numpy()[j,:,:audio_len[j]], axis= 0)) #求每一行的最大值索引
            label_all.extend(target.detach().cpu().numpy()[j,:target_len[j]])
            # print(f1_score(label_all,prob_all))
        # if i%100==0:
        #     print(np.sum(torch.argmax(logits, dim= 1).cpu().numpy()))
        #     print(np.sum(target.cpu().numpy()))

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))), 
                              acc="{:.04f}%".format(float(tacc*100 / (i + 1) / logits.shape[0])),
                              f1="{:.04f}%".format(float(f1_score(label_all,prob_all)*100)))
        batch_bar.update()

        ### Release memory
        del audio_fea, text_fea, target
        torch.cuda.empty_cache()
  
    batch_bar.close()
    tloss   /= len(train_loader)
    tacc    /= len(train_loader)*config['batch_size']
    tf1  = f1_score(label_all,prob_all)
    return tloss, tacc, tf1


def eval(model, dataloader):

    model.eval() # set model in evaluation mode
    vloss, vacc = 0, 0 # Monitoring loss and accuracy

    prob_all = []
    label_all = []

    batch_bar   = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    for i, (audio_fea, text_fea, target, audio_len, target_len) in enumerate(dataloader):
        ### Move Data to Device (Ideally GPU)
        audio_fea = audio_fea.to(device)
        text_fea  = text_fea.to(device)
        target = target.to(device)

        # makes sure that there are no gradients computed as we are not training the model now
        with torch.inference_mode(): 
            ### Forward Propagation
            logits  = model(audio_fea, text_fea).permute(0,2,1)
            ### Loss Calculation
            # for j in range(logits.shape[2]):
            #     if j==0:
            #         loss  = criterion(logits[:,:,j], target[:,j])
            #     else:
            #         loss  += criterion(logits[:,:,j], target[:,j])
            loss    = criterion(logits, target)
            # loss = criterion(torch.argmax(logits, dim=1),target)


        vloss   += loss.item()
        for j in range(logits.shape[0]):
            vacc += torch.mean(torch.argmax(logits[j,:,:audio_len[j]], dim= 0) == target[j,:target_len[j]], dtype=torch.float32).item()
            prob_all.extend(np.argmax(logits.detach().cpu().numpy()[j,:,:audio_len[j]], axis= 0)) #求每一行的最大值索引
            label_all.extend(target.detach().cpu().numpy()[j,:target_len[j]])

        # Do you think we need loss.backward() and optimizer.step() here?

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))), 
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1) / logits.shape[0])),
                              f1="{:.04f}%".format(float(f1_score(label_all,prob_all)*100)))

        batch_bar.update()
    
        ### Release memory
        del audio_fea, text_fea, target
        torch.cuda.empty_cache()

    batch_bar.close()
    vloss   /= len(val_loader)
    vacc    /= len(val_loader)*config['batch_size']
    vf1   = f1_score(label_all,prob_all)

    return vloss, vacc, vf1

best_acc = 0

for epoch in range(config['epochs']):

    print("\nEpoch {}/{}".format(epoch+1, config['epochs']))

    curr_lr                 = float(optimizer.param_groups[0]['lr'])
    train_loss, train_acc, train_f1   = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_f1   = eval(model, val_loader)
    test_loss, test_acc, test_f1   = eval(model, test_loader)
    scheduler.step()

    with open('./log_cross.txt','a') as f:
        f.write("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\tTrain F1 {:.04f}\t Learning Rate {:.07f}\n".format\
                (train_acc*100,train_loss, train_f1*100, curr_lr))
        f.write("\tVal Acc {:.04f}%\tVal Loss {:.04f}\tVal F1 {:.04f}\n".format(val_acc*100, val_loss, val_f1*100))
        f.write("\ttest Acc {:.04f}%\ttest Loss {:.04f}\ttest F1 {:.04f}\n".format(test_acc*100, test_loss, test_f1*100))
    print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\tTrain F1 {:.04f}\t Learning Rate {:.07f}".format(train_acc*100,\
                                                                                                       train_loss, train_f1*100, curr_lr))
    print("\tVal Acc {:.04f}%\tVal Loss {:.04f}\tVal F1 {:.04f}".format(val_acc*100, val_loss, val_f1*100))
    print("\ttest Acc {:.04f}%\ttest Loss {:.04f}\ttest F1 {:.04f}".format(test_acc*100, test_loss, test_f1*100))
    
    torch.save(model.state_dict(), 'model_cross.pkl')

    if val_acc>best_acc:
        torch.save(model.state_dict(), 'best_cross.pkl')
        best_acc = val_acc
    ### Log metrics at each epoch in your run 
    # Optionally, you can log at each batch inside train/eval functions 
    # (explore wandb documentation/wandb recitation)
    # wandb.log({'train_acc': train_acc*100, 'train_loss': train_loss, 
            #    'val_acc': val_acc*100, 'valid_loss': val_loss, 'lr': curr_lr})    
    # wandb.log({'train_acc': train_acc*100, 'train_loss': train_loss, 'lr': curr_lr})

    ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best

### Finish your wandb run
# run.finish()

# %%


# %%



