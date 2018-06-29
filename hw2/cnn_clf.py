"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
torchvision
"""
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as Data
import pickle
import numpy as np
from optparse import OptionParser
import json
from util import match_relation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)    # reproducible
np.random.seed(7)

# opts
op = OptionParser()
op.add_option("--process_name",
              dest="process_name", type=str,
              help="process name")

(opts, args) = op.parse_args()
if not opts.process_name:  op.error('process_name is not given')


######################## load data ########################


RESULT_PATH = "./results/proposed_answer_" + opts.process_name +  ".txt"
MODEL_PATH = "./model/model_" + opts.process_name + ".pkl"


x_train = np.load('./data/elmo_x_train.npy')
y_train = np.load('./data/y_train.npy')
x_test = np.load('./data/elmo_x_test.npy')


# x_train = torch.LongTensor(x_train)
x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)


split_prob = 0.1
n_valid = int(x_train.shape[0] * split_prob)
x_valid = x_train[-n_valid:, :]
x_train = x_train[:-n_valid, :]
y_valid = y_train[-n_valid:]
y_train = y_train[:-n_valid]


x_test = torch.FloatTensor(x_test)

# Hyper Parameters
n_epochs = 100              # train the training data n times, to save time, we just train 1 epoch
batch_size = 128
time_steps = x_train.shape[1]        
INPUT_SIZE = 28         # model input size / image width
LR = 0.001               # learning rate
# embed_dim = 300
embed_dim = 1024
n_classes = 19 # mnist classes/labels (0-9)
kernel_num = 300


torch_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)



class CNN_Text(nn.Module):    
    def __init__(self):
        super(CNN_Text, self).__init__()
        
        # self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)


        Ci = 1 #  input channel
        Co = kernel_num
        Ks = [2,3,4]

        # self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embed_dim)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, n_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        
        logit = self.fc1(x)  # (N, C)

        logit = F.log_softmax(logit, dim=1)

        return logit



model = CNN_Text()
model.cuda() 
print(model)

optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

def validate(model, x_valid, y_valid):
    model.eval()
    valid_output = model(x_valid.cuda())

    # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # 将操作放去 GPU
    pred_y = torch.max(valid_output, 1)[1].data.cpu().numpy()
    y_valid = y_valid.numpy()

    accuracy = float(sum(pred_y == y_valid)) / len(y_valid)

    return accuracy


# training and testing
loss = 0
n_valid_acc_no_change = 0
max_valid_acc = -1

for epoch in range(n_epochs):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        model.train()
        b_x, b_y = b_x.cuda(), b_y.cuda()
        # b_x = b_x.view(-1, time_steps)              # reshape x to (batch, time_step, input_size)

        # print (b_x)
        # print (b_y)
        # input()

        output = model(b_x)                               # model output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients


    # early stopping
    valid_acc = validate(model, x_valid, y_valid)
    if max_valid_acc < valid_acc:
        max_valid_acc = valid_acc
        n_valid_acc_no_change = 0
    else:
        n_valid_acc_no_change += 1

    if n_valid_acc_no_change >= 10:
        break

    
    print ("epoch: %d, loss: %.3f, valid_acc: %.3f"  % (epoch, loss.data.cpu().numpy(), valid_acc ))
    print ("-"*50)


torch.save(model, MODEL_PATH)  # 保存整个网络


# print 10 predictions from test data
test_output = model(x_test.cuda())

# pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # 将操作放去 GPU
pred_classes = torch.max(test_output, 1)[1].cuda().data.cpu().numpy()

pointer = 8001
with open(RESULT_PATH, 'w') as f:
    for i, pred_class in enumerate(pred_classes):
        relation = match_relation(pred_class)
        f.write("%d\t%s\n" % (i+pointer, relation))




