import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab
import numpy as np
# import random
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torch import optim

# random_seed = 42
# torch.manual_seed(random_seed)  # Sets the seed for generating random numbers. Returns a torch.Generator object.
# np.random.seed(random_seed)  # 난수 생성기를 초기화합니다.
# random.seed(random_seed)

train_data = IMDB(root='.data', split='train')
test_data = IMDB(root='.data', split='test')

train_data = to_map_style_dataset(train_data)
test_data = to_map_style_dataset(test_data)

tokenizer = get_tokenizer('basic_english')
vec = torchtext.vocab.GloVe(name='6B', dim=50)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    pre_text = []
    pre_labels = []
    pre_text = torch.tensor(pre_text)
    for (label, text) in batch:
        text = tokenizer(text)[:150]
        text += ["<pad>" for i in range(150 - len(text) if 150 > len(text) else 0)]  # text -> list
        pre_text = torch.cat([pre_text, vec.get_vecs_by_tokens(text)], dim=0)  # .cat: 같은 차원끼리만 합칠 수 있
        # print(pre_text.shape)  # torch.Size([150, 50]) -> +150 => 2차원
        pre_labels += [1 if label == 'pos' else 0]
    pre_text = pre_text.view(-1, 1, 150, 50)
    # print(pre_text.shape)  # torch.Size([50, 1, 150, 50])
    pre_labels = torch.FloatTensor(pre_labels)

    return pre_text, pre_labels


batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


# Model
class CNN(nn.Module):
    def __init__(self, batch, length, input_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.length = length
        self.batch = batch
        self.conv3 = nn.Conv2d(1, 100, (3, input_size), bias=True)
        self.conv4 = nn.Conv2d(1, 100, (4, input_size), bias=True)
        self.conv5 = nn.Conv2d(1, 100, (5, input_size), bias=True)
        self.Max3_pool = nn.MaxPool2d((length - 3 + 1, 1))
        self.Max4_pool = nn.MaxPool2d((length - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((length - 5 + 1, 1))
        self.linear1 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x1 = F.relu(self.conv3(x))
        # print(x1.shape)  # torch.Size([50, 100, 148, 1])
        x2 = F.relu(self.conv4(x))
        # print(x2.shape)  # torch.Size([50, 100, 147, 1])
        x3 = F.relu(self.conv5(x))
        # print(x3.shape)  # torch.Size([50, 100, 146, 1])

        # pooling
        x1 = self.Max3_pool(x1)
        # print(x1.shape)  # torch.Size([50, 100, 1, 1])
        x2 = self.Max4_pool(x2)
        # print(x2.shape)  # torch.Size([50, 100, 1, 1])
        x3 = self.Max5_pool(x3)
        # print(x3.shape)  # torch.Size([50, 100, 1, 1])

        x = torch.cat((x1, x2, x3), -1)
        # print(x.shape)  # torch.Size([50, 100, 1, 3])
        x = x.view(self.batch, 300)
        # print(x.shape)  # torch.Size([50, 300])
        x = self.dropout(x)
        x = F.sigmoid(self.linear1(x))

        return x


net = CNN(50, 150, 50)

# Train
criterion = nn.BCELoss()  # loss
optimizer = optim.Adam(net.parameters(), lr=1e-6)
# acc = []
for epoch in range(300):
    losses = 0
    for (text, labels) in train_loader:
        # print(text.shape)  # torch.Size([50, 1, 150, 50])
        # print(net(text).shape)  # torch.Size([50, 1])
        predictions = net(text).squeeze()
        # print(predictions.shape)  # torch.Size([50])
        # print(labels.shape)  # torch.Size([50])
        loss = criterion(predictions, labels)
        # print(loss.shape)  # torch.Size([])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses = losses + loss.item()
    print(f'Epoch{epoch + 1},training loss: {losses/(len(train_data)/batch_size)}')

# Test
# 코드 블럭을 with torch.no_grad 로 래핑해 required_grad=True 속성이 명시된 autograd 가 텐서의 추적 기록에 남기지 않게
with torch.no_grad():
    num_correct = 0
    net.eval()
    for (text, labels) in test_loader:
        output = net(text)
        # print(output.shape)  # torch.Size([50, 1])
        pred = torch.round(output.squeeze())
        # print(pred.shape)  # torch.Size([50])

        # compare predictions to true label
        correct_tensor = pred.eq(labels.view_as(pred))
        # print(correct_tensor.shape)  # torch.Size([50])
        # print(correct_tensor.numpy().shape)  # (50,)
        correct = np.squeeze(correct_tensor.numpy())
        # print(correct.shape)  # (50,)
        num_correct += np.sum(correct)

    test_acc = num_correct / (len(test_loader)*batch_size)
    print("Test accuracy: {:.3f}".format(test_acc))




