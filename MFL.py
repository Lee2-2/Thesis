import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn, optim
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import copy
import pickle
import csv
import time
import math
import datetime


# torch       == 1.13.1
# torchvision == 0.14.1
# pandas      == 1.4.4
# numpy       == 1.22.4
# matplotlib  == 3.7.1
# pip install torch==1.13.1 torchvision==0.14.1 pandas==1.4.4 numpy==1.22.4 matplotlib==3.7.1

def fix_seed(seed): #random 分布資料
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


SEED = 8  # 如果使用相同的seed( )值，则每次生成的隨機数都相同
fix_seed(SEED)

start_time = datetime.datetime.now()
print(start_time)
print("cifer10_DBSCAN_client++")


class Argments():
    def __init__(self):
        self.batch_size = 256  # 每次抓取的size 越大需要越多運算空間
        self.test_batch = 1000
        self.global_epochs = 70  # 設定global_epochs
        self.local_epochs = 2  # 設定local_epochs
        self.lr = None
        self.momentum = 0.9
        self.weight_decay = 10 ** -4.0
        self.clip = 20.0
        self.partience = 500
        self.worker_num = 10  # 決定有幾個客戶端
        self.sample_num = 10 # 決定抓取幾個客戶端
        self.cluster_list = [2, 3, 4, 5]  # 聚類標籤
        self.cluster_num = None
        self.turn_of_cluster_num = [0,3,50,75]  # 在第幾globe epoch進行交換
        # self.turn_of_replacement_model = list(range(0,self.global_epochs,10))
        self.turn_of_replacement_model = list(range(self.global_epochs))
        self.unlabeleddata_size = 1000
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.clustering_algorithm = "dbscan"  # "dbscan" use DBSCAN algorithm "kmeans" use k-means algorithm

        self.alpha_label = 0.1
        self.alpha_size = 10


args = Argments()

# Tuned value
lr = 2

lr_list = []
lr_list.append(10 ** -3.0)
lr_list.append(10 ** -2.5)
lr_list.append(10 ** -2.0)
lr_list.append(10 ** -1.5)
lr_list.append(10 ** -1.0)
lr_list.append(10 ** -0.5)
lr_list.append(10 ** 0.0)
lr_list.append(10 ** 0.5)

args.lr = 0.01


class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):  # 是一個建構函數，它創建了一個 LocalDataset 的實例。它接受一個可選的 transform 參數，這個參數可以是一個函數，用於對數據進行轉換。
        self.transform = transform
        self.data = []
        self.label = []

    def __len__(self):  # 是一個特殊方法，它返回數據集中的樣本數量
        return len(self.data)

    def __getitem__(self,
                    idx):  # 是另一個特殊方法，用於根據給定的索引 idx 檢索數據集中的樣本。該方法返回一個數據/標籤對 (out_data, out_label)。如果指定了 transform 函數，則該函數將被用於對 out_data 進行轉換。
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label


class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset,
                 transform=None):  # 它接受兩個參數，subset 是一個 PyTorch 數據集對象，表示從中提取樣本的子集，transform 是一個可選的函數，用於對數據進行轉換。
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class GlobalDataset(torch.utils.data.Dataset):
    def __init__(self, federated_dataset, transform=None):
        self.transform = transform
        self.data = []
        self.label = []
        for dataset in federated_dataset:
            for (data, label) in dataset:
                self.data.append(data)
                self.label.append(label)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label

    def __len__(self):
        return len(self.data)


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        self.target = None

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = 'unlabeled'
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label

    def __len__(self):
        return len(self.data)


def get_dataset(Centralized=True, unlabeled_data=False):
    #####手動隱蔽#####
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    from sklearn.model_selection import train_test_split
    col_names = [' Protocol', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
                 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
                 ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max',
                 ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
                 ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total',
                 ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
                 ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
                 ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
                 ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
                 ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
                 ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
                 ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
                 ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
                 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward',
                 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max',
                 ' Idle Min', 'Label']  # 80
    feature_cols = [' Protocol', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
                    'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
                    ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                    'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
                    ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
                    ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
                    ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
                    'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
                    ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
                    ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
                    ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
                    ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
                    ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
                    ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
                    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets',
                    ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
                    ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
                    ' Idle Std', ' Idle Max', ' Idle Min']  # 79
    label_cols = ['Label']
    all_trainset = pd.DataFrame()
    ###
    from pathlib import Path
    files = Path('/home/icmnlab/LEE/dataset/train/HR+LR/').glob('*.csv')  # get all csvs in your dir.
    # files = Path('/home/icmnlab/LEE/dataset/train/jj/').glob('*.csv')  # get all csvs in your dir.
    for file in files:
        print(file)
        # all_trainset = pd.read_csv(file, header=0, names=col_names, low_memory=False)
        temp_df = pd.read_csv(file, header=0, names=col_names, low_memory=False)
        all_trainset = pd.concat([all_trainset, temp_df], ignore_index=True)

        # all_trainset = pd.read_csv('/home/oem/fast4_0805.csv', header=0, names=col_names, low_memory=False)

    all_trainset.replace([np.inf, -np.inf], 0, inplace=True)  # remove infinity
    all_trainset.dropna(inplace=True)  # remove space
    train = all_trainset[feature_cols].values
    label = all_trainset[label_cols].values

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    train = ss.fit_transform(train)

    # 進行Label encoding編碼
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(label)
    label = encoder.transform(label)

    # one hot
    data_dum = pd.get_dummies(all_trainset[label_cols])
    label = pd.DataFrame(data_dum)
    print(label)
    contains_nine = (label == 9)

    ########################

    ########################################################################
    # 切割資料集
    from torch.utils.data.sampler import SubsetRandomSampler
    test_split = 0.2
    shuffle_dataset = True
    random_seed = 1234
    dataset_size = len(all_trainset)

    train_dataset_size = len(train)
    label_dataset_size = len(label)

    indices = list(range(dataset_size))
    train_indices = list(range(train_dataset_size))
    test_indices = list(range(label_dataset_size))
    # count out split size
    split = int(np.floor(test_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_data_indices, test_data_indices = train_indices[split:], train_indices[:split]

    # def to_categorical(y, num_classes):
    #     """ 1-hot encodes a tensor """
    #     return np.eye(num_classes, dtype='float32')[y]

    trainX = train[::].reshape(train.shape[0], 1, 78).astype('float32')  # 79是看feature_cols有幾個
    labelX = label.values.reshape(label.shape[0], 1, 1).astype('float32')
    all_train_data, all_test_data, all_train_label, all_test_label = train_test_split(trainX, labelX, test_size=0.3, random_state=None)
    all_train_data = np.array(all_train_data)
    all_test_data = np.array(all_test_data)
    all_train_label = np.array(all_train_label)
    all_test_label = np.array(all_test_label)

    ## Data size heterogeneity
    data_proportions = np.random.dirichlet(np.repeat(args.alpha_size, args.worker_num))
    train_data_proportions = np.array([0 for _ in range(args.worker_num)])
    test_data_proportions = np.array([0 for _ in range(args.worker_num)])
    for i in range(len(data_proportions)):
        if i == (len(data_proportions) - 1):
            train_data_proportions = train_data_proportions.astype('int64')
            test_data_proportions = test_data_proportions.astype('int64')
            train_data_proportions[-1] = len(all_train_data) - np.sum(train_data_proportions[:-1])
            test_data_proportions[-1] = len(all_test_data) - np.sum(test_data_proportions[:-1])
        else:
            train_data_proportions[i] = (data_proportions[i] * len(all_train_data))
            test_data_proportions[i] = (data_proportions[i] * len(all_test_data))
    min_size = 0
    K = 11

    '''
    label_list = np.arange(10)
    np.random.shuffle(label_list)
    '''
    label_list = list(range(K))

    ## Data distribution heterogeneity
    while min_size < 11:
        idx_train_batch = [[] for _ in range(args.worker_num)]
        idx_test_batch = [[] for _ in range(args.worker_num)]
        for k in label_list:
            proportions_train = np.random.dirichlet(np.repeat(args.alpha_label, args.worker_num))
            proportions_test = copy.deepcopy(proportions_train)
            idx_k_train = np.where(all_train_label == k)[0]
            idx_k_test = np.where(all_test_label == k)[0]
            np.random.shuffle(idx_k_train)
            np.random.shuffle(idx_k_test)
            ## Balance (train)
            proportions_train = np.array([p * (len(idx_j) < train_data_proportions[i]) for i, (p, idx_j) in
                                          enumerate(zip(proportions_train, idx_train_batch))])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k_train)).astype(int)[:-1]
            idx_train_batch = [idx_j + idx.tolist() for idx_j, idx in
                               zip(idx_train_batch, np.split(idx_k_train, proportions_train))]

            ## Balance (test)
            proportions_test = np.array([p * (len(idx_j) < test_data_proportions[i]) for i, (p, idx_j) in
                                         enumerate(zip(proportions_test, idx_test_batch))])
            proportions_test = proportions_test / proportions_test.sum()
            proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
            idx_test_batch = [idx_j + idx.tolist() for idx_j, idx in
                              zip(idx_test_batch, np.split(idx_k_test, proportions_test))]

            min_size = min([len(idx_j) for idx_j in idx_train_batch])

    federated_trainset = []
    federated_testset = []
    for i in range(args.worker_num):
        ## create trainset
        data = [all_train_data[idx] for idx in idx_train_batch[i]]
        label = [all_train_label[idx] for idx in idx_train_batch[i]]
        federated_trainset.append(LocalDataset())
        federated_trainset[-1].data = data
        federated_trainset[-1].label = label

        ## create testset
        data = [all_test_data[idx] for idx in idx_test_batch[i]]
        label = [all_test_label[idx] for idx in idx_test_batch[i]]
        federated_testset.append(LocalDataset())
        federated_testset[-1].data = data
        federated_testset[-1].label = label

    ## split trainset
    federated_valset = [None] * args.worker_num
    for i in range(args.worker_num):
        n_samples = len(federated_trainset[i])
        if n_samples == 1:
            train_subset = federated_trainset[i]
            val_subset = copy.deepcopy(federated_trainset[i])
        else:
            train_size = int(len(federated_trainset[i]) * 0.8)
            val_size = n_samples - train_size
            train_subset, val_subset = torch.utils.data.random_split(federated_trainset[i], [train_size, val_size])

        federated_trainset[i] = DatasetFromSubset(train_subset)
        federated_valset[i] = DatasetFromSubset(val_subset)

    ## show data distribution
    H = 5
    W = 4
    fig, axs = plt.subplots(H, W, figsize=(20, 15))
    x = np.arange(0, 11)
    for i, (trainset, valset, testset) in enumerate(zip(federated_trainset, federated_valset, federated_testset)):
        bottom = [0] * 11
        count = [0] * 11
        for _, label in trainset:
            count[label[0][0].astype(int)] += 1
        axs[int(i / W), i % W].bar(x, count, bottom=bottom)
        for j in range(len(count)):
            bottom[j] += count[j]
        count = [0] * 11
        for _, label in valset:
            count[label[0][0].astype(int)] += 1
        axs[int(i / W), i % W].bar(x, count, bottom=bottom)
        for j in range(len(count)):
            bottom[j] += count[j]
        count = [0] * 11
        for _, label in testset:
            count[label[0][0].astype(int)] += 1
        axs[int(i / W), i % W].bar(x, count, bottom=bottom)
        # axs[int(i/W), i%W].title("worker{}".format(i+1), fontsize=12, color = "green")

    plt.show()

    ## get global dataset
    if Centralized:
        global_trainset = GlobalDataset(federated_trainset)
        global_valset = GlobalDataset(federated_valset)
        global_testset = GlobalDataset(federated_testset)

        # show_cifer(global_trainset.data,global_testset.label, cifar10_labels)

        global_trainset.transform = transform_train
        global_valset.transform = transform_test
        global_testset.transform = transform_test

        global_trainloader = torch.utils.data.DataLoader(global_trainset, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=2)
        global_valloader = torch.utils.data.DataLoader(global_valset, batch_size=args.test_batch, shuffle=False,
                                                       num_workers=2)
        global_testloader = torch.utils.data.DataLoader(global_testset, batch_size=args.test_batch, shuffle=False,
                                                        num_workers=2)

    ## set transform
    for i in range(args.worker_num):
        federated_trainset[i].transform = transform_train
        federated_valset[i].transform = transform_test
        federated_testset[i].transform = transform_test

    # if Centralized and unlabeled_data:
    #     return federated_trainset, federated_valset, federated_testset, global_trainloader, global_valloader, global_testloader, unlabeled_dataset
    if Centralized:
        return federated_trainset, federated_valset, federated_testset, global_trainloader, global_valloader, global_testloader
    # elif unlabeled_data:
    #     return federated_trainset, federated_valset, federated_testset, unlabeled_dataset
    else:
        return federated_trainset, federated_valset, federated_testset


federated_trainset, federated_valset, federated_testset, global_trainloader, global_valloader, global_testloader  = get_dataset(unlabeled_data=True)

total = [0, 0, 0]
for i in range(args.worker_num):
    total[0] += len(federated_trainset[i])
    total[1] += len(federated_valset[i])
    total[2] += len(federated_testset[i])
print("total = ", total)


from collections import Counter
sums_1_to_8 = []
sums_9_to_10 = []

for i in range(args.worker_num):
    print("Worker federated", federated_trainset[i])
    labels = [label for _, label in federated_trainset[i]]
    # print("Labels ", labels)
    # 將 numpy.ndarray 標籤轉換為 Python 基本設具類型
    labels = [int(label) if isinstance(label, np.ndarray) else label for label in labels]
    # 統計每個標籤的數量
    label_counts = Counter(labels)

    sum_1_to_8 = 0
    sum_9_to_10 = 0

    # 每個標籤的數量
    for label, count in label_counts.items():
        print(f"Label: {label}, Count: {count}")
        if 1 <= label <= 8:
            sum_1_to_8 += count
        elif 9 <= label <= 10:
            sum_9_to_10 += count

    sums_1_to_8.append(sum_1_to_8)
    sums_9_to_10.append(sum_9_to_10)

client_model=[]
for worker_index in range(args.worker_num):
    print(f"Worker {worker_index}: Sum of labels 1-8 = {sums_1_to_8[worker_index]}, Sum of labels 9-10 = {sums_9_to_10[worker_index]}")
    if sums_9_to_10[worker_index] > 0:
        client_model.append(random.choice([1,2,3 ,4,5]))
    else:
        client_model.append(random.choice([1,2,3, 4,5]))

    print("client model: ", client_model)


class simpleLSTM(nn.Module): #par = 342 #acc=91
    def __init__(self, input_size=78, hidden_size=32, output_size=11): #1
        super(simpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        #self.linear1 = nn.Linear(hidden_size, output_size)  # Adjust out_features
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))
         # Pass through LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        import torch.nn.functional as F
        out = F.softmax(out, dim=-1)
        return out

class LSTM4(nn.Module): # acc=70
    def __init__(self, num_classes=11, input_size=78, hidden_size=8, num_layers=1, dropout_prob=0.5):
        super(LSTM4, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        x = x.view(x.size(0), -1, x.size(-1))
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        import torch.nn.functional as F
        out = F.softmax(out, dim=-1)

        return out


class LSTM3(nn.Module): # acc=70
    def __init__(self, num_classes=11, input_size=78, hidden_size=16, num_layers=1, dropout_prob=0.5):
        super(LSTM3, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        x = x.view(x.size(0), -1, x.size(-1))
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        import torch.nn.functional as F
        out = F.softmax(out, dim=-1)

        return out

class LSTM2(nn.Module): # acc=70
    def __init__(self, num_classes=11, input_size=78, hidden_size=32, num_layers=1, dropout_prob=0.5):
        super(LSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        x = x.view(x.size(0), -1, x.size(-1))
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        import torch.nn.functional as F
        out = F.softmax(out, dim=-1)

        return out


class LSTM1(nn.Module): # acc=70
    def __init__(self, num_classes=11, input_size=78, hidden_size=64, num_layers=1, dropout_prob=0.5):
        super(LSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        x = x.view(x.size(0), -1, x.size(-1))
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        import torch.nn.functional as F
        out = F.softmax(out, dim=-1)

        return out


class LSTM0(nn.Module): # acc=70
    def __init__(self, num_classes=11, input_size=78, hidden_size=128, num_layers=1, dropout_prob=0.5):
        super(LSTM0, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device).requires_grad_()
        x = x.view(x.size(0), -1, x.size(-1))
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        import torch.nn.functional as F
        out = F.softmax(out, dim=-1)

        return out


from concurrent.futures import ThreadPoolExecutor

from scipy.spatial import KDTree

class DBSCAN(object):
    def __init__(self, eps, min_samples, batch_size=10000):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.batch_size = batch_size
        self.kd_tree = None

    def fit_predict(self, features):
        features = torch.tensor(features, device=args.device, dtype=torch.float32)
        n_samples = features.shape[0]
        self.labels_ = -torch.ones(n_samples, device=args.device, dtype=torch.int32)

        # 构建 KD 树
        self.kd_tree = KDTree(features.cpu().numpy().reshape(n_samples, -1))

        cluster_id = 0
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_features = features[start:end]
            batch_labels = self.labels_[start:end]

            neighbors_batch = self._find_neighbors(batch_features)
            # print("neighbors_batch", neighbors_batch)
            for i in range(batch_features.shape[0]):
                # print("i",i)
                if batch_labels[i] != -1:
                    continue

                neighbors = neighbors_batch[i]
                # print("neighbors", neighbors)
                if neighbors.shape[0] < self.min_samples:
                    batch_labels[i] = -1
                else:
                    # print("expend")
                    self._expand_cluster(batch_features, i, cluster_id, neighbors, batch_labels)
                    # print("expend2")
                    cluster_id += 1

            self.labels_[start:end] = batch_labels

        return self.labels_.cpu().numpy()

    def _find_neighbors(self, features):
        features_np = features.cpu().numpy().reshape(features.shape[0], -1)
        distances, indices = self.kd_tree.query(features_np, k=self.min_samples)  # 查询 k 个最近邻居

        neighbors = torch.tensor(indices, device=args.device, dtype=torch.int64)

        return neighbors

    def _expand_cluster(self, features, i, cluster_id, neighbors, labels):
        labels[i] = cluster_id
        queue = list(neighbors.cpu().numpy())

        while queue:
            neighbor = queue.pop(0)

            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id

            if labels[neighbor] != cluster_id:
                continue

            labels[neighbor] = cluster_id
            new_neighbors = self._find_neighbors(features[neighbor].unsqueeze(0))

            if new_neighbors.shape[0] >= self.min_samples:  # Check if enough neighbors are found
                queue.extend(new_neighbors.cpu().numpy())



class Server():
    # 初始化Server類別
    def __init__(self, global_trainloader):
        self.cluster = None  # 用於存放分群資訊的變數
        self.models = {}  # 用於存放Worker模型的變數
        self.models_replace= {}
        self.global_trainloader=global_trainloader

        # 創建無標籤數據加載器，用於處理無標籤數據集
        # self.unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
        #                                                         shuffle=False, num_workers=2)

    def create_worker(self, federated_trainset, federated_valset, federated_testset, client_best_model):
        workers = []
        for i in range(args.worker_num):
            workers.append(
                Worker(i, federated_trainset[i], federated_valset[i], federated_testset[i], client_best_model[i]))
        return workers

    # 隨機抽樣一定數量的Worker
    def sample_worker(self, workers):
        sample_worker = []
        sample_worker_num = random.sample(range(args.worker_num), args.sample_num)
        for i in sample_worker_num:
            sample_worker.append(workers[i])
        return sample_worker

    def collect_model(self, workers):
        self.models = [None] * args.worker_num
        for worker in workers:
            self.models[worker.id] = copy.deepcopy(worker.local_model)

    def send_model(self, workers):
        for worker in workers:
            worker.local_model = copy.deepcopy(self.models[worker.id])
            worker.other_model = copy.deepcopy(self.models[worker.other_model_id])
            worker.third_model = copy.deepcopy(self.models[worker.third_model_id])
            worker.four_model = copy.deepcopy(self.models[worker.four_model_id])
            worker.five_model = copy.deepcopy(self.models[worker.five_model_id])


    def return_model(self, workers):
        for worker in workers:
            worker.local_model = copy.deepcopy(self.models[worker.local_model_id])
            worker.local_model_id = worker.id
        del self.models

    def aggregate_model5(self, workers):
        new_params = []  # params參數 用於存儲聚合後的參數
        train_model_id = []  # 用於存儲訓練模型的 ID
        train_model_id_count = []  # 用於存儲每個模型訓練的次數
        # 遍歷所有Worker
        for worker in workers:
            # 獲取Worker本地模型參數
            worker_state = worker.local_model.state_dict()
            # 將本地模型參數加入到 new_params 中
            if worker.id in train_model_id:
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]
            # 獲取工作者的其他模型參數
            worker_state = worker.other_model.state_dict()

            # 將其他模型參數加入到 new_params 中
            if worker.other_model_id in train_model_id:
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.other_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

            worker_state = worker.third_model.state_dict()
            # 將第三個模型參數加入到 new_params 中
            if worker.third_model_id in train_model_id:
                i = train_model_id.index(worker.third_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.third_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.third_model_id)

                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

            worker_state = worker.four_model.state_dict()
            # 將第4個模型參數加入到 new_params 中
            if worker.four_model_id in train_model_id:
                i = train_model_id.index(worker.four_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.four_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.four_model_id)

                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

            worker_state = worker.five_model.state_dict()
            # 將第4個模型參數加入到 new_params 中
            if worker.five_model_id in train_model_id:
                i = train_model_id.index(worker.five_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.five_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.five_model_id)

                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

        # 計算聚合後的模型參數並加載到 self.models 中

        for i, model_id in enumerate(train_model_id):
            for key in new_params[i].keys():
                new_params[i][key] = new_params[i][key] / train_model_id_count[i]
            self.models[model_id].load_state_dict(new_params[i])
            # worker.replace_model_id = model_id



    def aggregate_model4(self, workers):
        new_params = []  # params參數 用於存儲聚合後的參數
        train_model_id = []  # 用於存儲訓練模型的 ID
        train_model_id_count = []  # 用於存儲每個模型訓練的次數
        # 遍歷所有Worker
        for worker in workers:
            # 獲取Worker本地模型參數
            worker_state = worker.local_model.state_dict()
            # 將本地模型參數加入到 new_params 中
            if worker.id in train_model_id:
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]
            # 獲取工作者的其他模型參數
            worker_state = worker.other_model.state_dict()

            # 將其他模型參數加入到 new_params 中
            if worker.other_model_id in train_model_id:
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.other_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

            worker_state = worker.third_model.state_dict()
            # 將第三個模型參數加入到 new_params 中
            if worker.third_model_id in train_model_id:
                i = train_model_id.index(worker.third_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.third_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.third_model_id)

                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

            worker_state = worker.four_model.state_dict()
            # 將第4個模型參數加入到 new_params 中
            if worker.four_model_id in train_model_id:
                i = train_model_id.index(worker.four_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.four_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.four_model_id)

                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

        # 計算聚合後的模型參數並加載到 self.models 中

        for i, model_id in enumerate(train_model_id):
            for key in new_params[i].keys():
                new_params[i][key] = new_params[i][key] / train_model_id_count[i]
            self.models[model_id].load_state_dict(new_params[i])


    def aggregate_model3(self, workers):
        new_params = []  # params參數 用於存儲聚合後的參數
        train_model_id = []  # 用於存儲訓練模型的 ID
        train_model_id_count = []  # 用於存儲每個模型訓練的次數
        # 遍歷所有Worker
        for worker in workers:
            # 獲取Worker本地模型參數
            worker_state = worker.local_model.state_dict()
            # 將本地模型參數加入到 new_params 中
            if worker.id in train_model_id:
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]
            # 獲取工作者的其他模型參數
            worker_state = worker.other_model.state_dict()

            # 將其他模型參數加入到 new_params 中
            if worker.other_model_id in train_model_id:
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.other_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]

            worker_state = worker.third_model.state_dict()
            # 將第三個模型參數加入到 new_params 中
            if worker.third_model_id in train_model_id:
                i = train_model_id.index(worker.third_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.third_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.third_model_id)

                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]


        # 計算聚合後的模型參數並加載到 self.models 中

        for i, model_id in enumerate(train_model_id):
            for key in new_params[i].keys():
                new_params[i][key] = new_params[i][key] / train_model_id_count[i]
            self.models[model_id].load_state_dict(new_params[i])
            # worker.replace_model_id = model_id
    def aggregate_model2(self, workers):
        new_params = []  # params參數 用於存儲聚合後的參數
        train_model_id = []  # 用於存儲訓練模型的 ID
        train_model_id_count = []  # 用於存儲每個模型訓練的次數
        # 遍歷所有Worker
        for worker in workers:
            # 獲取Worker本地模型參數
            worker_state = worker.local_model.state_dict()
            # 將本地模型參數加入到 new_params 中
            if worker.id in train_model_id:
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]
            # 獲取工作者的其他模型參數
            worker_state = worker.other_model.state_dict()

            # 將其他模型參數加入到 new_params 中
            if worker.other_model_id in train_model_id:
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] += worker_state[key]
                train_model_id_count[i] += 1
            else:
                new_params.append(OrderedDict())
                train_model_id.append(worker.other_model_id)
                train_model_id_count.append(1)
                i = train_model_id.index(worker.other_model_id)
                for key in worker_state.keys():
                    new_params[i][key] = worker_state[key]


        # 計算聚合後的模型參數並加載到 self.models 中

        for i, model_id in enumerate(train_model_id):
            for key in new_params[i].keys():
                new_params[i][key] = new_params[i][key] / train_model_id_count[i]
            self.models[model_id].load_state_dict(new_params[i])
            # worker.replace_model_id = model_id


    def clustering(self, workers):
        # 如果聚類數目為 1，則將所有工作者指定為同一類別
        if args.cluster_num == 1:
            pred = [0] * len(workers)
            worker_id_list = []
            for worker in workers:
                worker_id_list.append(worker.id)
        else:
            with torch.no_grad():
                worker_softmax_targets = [[] for i in range(len(workers))]
                worker_id_list = []
                count = 0
                # 遍歷所有模型，獲取它們在無標籤數據集上的輸出
                for i, model in enumerate(self.models):
                    # print("j",i)
                    if model == None:
                        pass
                    else:
                        model = model.to(args.device)
                        model.eval()
                        for batch_idx, (data, _) in enumerate(self.global_trainloader):
                            data = data.to(args.device)
                            # output = model(data).to('cpu').detach().numpy()
                            output = model(data).detach().cpu().numpy()
                            worker_softmax_targets[count].append(output)

                        shapes = [output.shape for output in worker_softmax_targets[count]]

                        def pad_to_shape(arr, target_shape):
                            result = np.zeros(target_shape, dtype=arr.dtype)
                            slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
                            result[slices] = arr[slices]
                            return result

                        target_shape = (256, 11)  # Set your desired shape here
                        padded_outputs = [pad_to_shape(output, target_shape) for output in
                                          worker_softmax_targets[count]]
                        worker_softmax_targets[count] = np.array(padded_outputs)

                        # for item in worker_softmax_targets[count]:

                        # print(item.shape)
                        worker_softmax_targets[count] = np.array(worker_softmax_targets[count])
                        model = model.to('cpu')
                        worker_id_list.append(i)
                        count += 1

                worker_softmax_targets = np.array(worker_softmax_targets)
                # 根據指定的聚類算法進行聚類
                if args.clustering_algorithm == "kmeans":
                    kmeans = KMeans(n_clusters=args.cluster_num)
                    pred = kmeans.fit_predict(worker_softmax_targets)
                elif args.clustering_algorithm == "dbscan":
                    # print("Running DBSCAN...")
                    dbscan = DBSCAN(eps=0.75, min_samples=5)
                    # print("enter dbscan")
                    pred = dbscan.fit_predict(worker_softmax_targets)
                    # print("dbscan fit predict")
        # 將聚類結果存儲到 self.cluster 中
        self.cluster = {}

        for i, cls in enumerate(pred):
            if cls not in self.cluster:
                self.cluster[cls] = []
            self.cluster[cls].append(worker_id_list[i])
        # 為每個Worker指定聚類結果
        counter = 1
        for worker in workers:
            idx = (worker.id)
            worker.cluster_num = pred[idx]
            # print("worker.cluster_num", counter ," = " , worker.cluster_num )
            counter += 1

    def decide_other_model(self, workers):
        for worker in workers:
            cls = worker.cluster_num  # 獲取工作者所在的聚類

            # 如果聚類中只有一個工作者，則從所有工作者中隨機選擇一個作為其他模型
            if len(self.cluster[cls]) == 1:
                while True:
                    other_worker = random.choice(workers)
                    other_model_id = other_worker.id
                    third_worker = random.choice(workers)
                    third_model_id = third_worker.id
                    four_worker = random.choice(workers)
                    four_model_id = four_worker.id
                    five_worker = random.choice(workers)
                    five_model_id = five_worker.id
                    # 確保選擇的其他模型不是自己
                    if worker.id != other_model_id:
                        break
                    if worker.id != third_model_id:
                        break
                    if worker.id != four_model_id:
                        break
                    if worker.id != five_model_id:
                        break
            else:
                while True:
                    other_model_id = random.choice(self.cluster[cls])
                    third_model_id = random.choice(self.cluster[cls])
                    four_model_id = random.choice(self.cluster[cls])
                    five_model_id = random.choice(self.cluster[cls])
                    if worker.id != other_model_id:
                        break
                    if worker.id != third_model_id:
                        break
                    if worker.id != four_model_id:
                        break
                    if worker.id != five_model_id:
                        break
            # print("other_model_id", other_model_id, "third_model_id", third_model_id)

            # 將選擇的其他模型 ID 賦值給 worker 的 other_model_id 屬性
            # print("other_model_id", other_model_id, "third_model_id", third_model_id)
            worker.other_model_id = other_model_id
            worker.third_model_id = third_model_id
            worker.four_model_id = four_model_id
            worker.five_model_id = five_model_id



class Worker():
    # 初始化 Worker 物件
    def __init__(self, i, trainset, valset, testset, best_model):
        self.id = i
        self.cluster_num = None
        # 定義訓練、驗證、測試資料的 DataLoader
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=2)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=2)
        # 根據 best_model 參數選擇不同的 VGG 模型
        if best_model == 1:
            self.local_model = LSTM1()
        elif best_model == 2:
            self.local_model = LSTM2()
        elif best_model == 3:
            self.local_model = LSTM3()
        elif best_model == 4:
            self.local_model = LSTM0()
        elif best_model == 5:
            self.local_model = LSTM4()


        self.local_model_id = i
        self.replace_model = None
        self.replace_model_id = None
        self.other_model = None
        self.other_model_id = None
        self.third_model = None
        self.third_model_id = None
        self.four_model = None
        self.four_model_id = None
        self.five_model = None
        self.five_model_id = None
        # 記錄訓練集和測試集的資料數量
        self.train_data_num = len(trainset)
        self.test_data_num = len(testset)

    # 本地訓練函數
    # 本地訓練函數
    def local_train5(self):
        # 將模型移到指定的設備上
        self.local_model = self.local_model.to(args.device)
        self.other_model = self.other_model.to(args.device)
        self.third_model = self.third_model.to(args.device)
        self.four_model = self.four_model.to(args.device)
        self.five_model = self.five_model.to(args.device)

        # 定義優化器
        local_optimizer = optim.SGD(self.local_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        other_optimizer = optim.SGD(self.other_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        third_optimizer = optim.SGD(self.third_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        four_optimizer = optim.SGD(self.four_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        five_optimizer = optim.SGD(self.five_model.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        # 設定模型為訓練模式
        self.local_model.train()
        self.other_model.train()
        self.third_model.train()
        self.four_model.train()
        self.five_model.train()

        y_true_train = []
        y_pred_train = []

        for epoch in range(args.local_epochs):
            running_loss = 0.0
            correct = 0
            count = 0
            # 讀取訓練資料
            for (data, labels) in self.trainloader:
                torch.autograd.set_detect_anomaly(True)
                # 將資料和標籤轉為變量並移到指定的設備上
                data, labels = Variable(data), Variable(labels)
                data, labels = data.to(args.device), labels.to(args.device)
                # 清空優化器的梯度
                local_optimizer.zero_grad()
                other_optimizer.zero_grad()
                third_optimizer.zero_grad()
                four_optimizer.zero_grad()
                five_optimizer.zero_grad()
                # 進行前向傳播
                local_outputs = self.local_model(data)
                other_outputs = self.other_model(data)
                third_outputs = self.third_model(data)
                four_outputs = self.four_model(data)
                five_outputs = self.five_model(data)

                labels = labels.to(torch.long)
                labels = labels.squeeze()
                # print("Modified target labels /shape:", labels.shape)
                # labels = labels.long()

                # print("Labels Shape:", labels.shape)
                if len(labels.shape) > 1:
                    labels = torch.argmax(labels, dim=1).long()


                # train local_model訓練本地模型
                ce_loss = args.criterion_ce(local_outputs, labels)  # 計算交叉熵損失(交叉熵損失（用於度量模型對目標標籤的預測能力）)
                kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                            F.softmax(Variable(other_outputs),
                                                      dim=1))  # 計算 KL 散度損失(用於度量兩個模型輸出之間的相似性)
                kl_loss2 = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                            F.softmax(Variable(third_outputs),
                                                      dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                             F.softmax(Variable(four_outputs),
                                                       dim=1))
                kl_loss4 = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                             F.softmax(Variable(five_outputs),
                                                       dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 + kl_loss3 +kl_loss4)/4  # 總損失為交叉熵損失與 KL 散度損失之和
                running_loss = running_loss+loss.item()
                predicted = torch.argmax(local_outputs, dim=1)  # 預測類別
                correct += (predicted == labels).sum().item()  # 計算正確預測的數量
                count += len(labels)  # 累計資料數量

                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())



                loss.backward(retain_graph=True)  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)  # 梯度裁剪
                local_optimizer.step()  # 更新本地模型參數
                # train other_model 訓練其他模型
                ce_loss = args.criterion_ce(other_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                            F.softmax(Variable(third_outputs), dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                             F.softmax(Variable(four_outputs), dim=1))
                kl_loss4 = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                             F.softmax(Variable(five_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 +kl_loss3 +kl_loss4)/4 # 總損失為交叉熵損失與 KL 散度損失之和
                loss.backward(retain_graph=True)  # 反向傳播計算梯度                running_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.other_model.parameters(), args.clip)  # 梯度裁剪
                other_optimizer.step()  # 更新其他模型參數

                # train third_model 訓練其他模型
                ce_loss = args.criterion_ce(third_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                             F.softmax(Variable(other_outputs), dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                             F.softmax(Variable(four_outputs), dim=1))
                kl_loss4 = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                             F.softmax(Variable(five_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 + kl_loss3+kl_loss4)/4  # 總損失為交叉熵損失與 KL 散度損失之和

                loss.backward()  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.third_model.parameters(), args.clip)  # 梯度裁剪
                third_optimizer.step()  # 更新其他模型參數

                # train four_model 訓練其他模型
                ce_loss = args.criterion_ce(four_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(four_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(four_outputs, dim=1),
                                             F.softmax(Variable(other_outputs), dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(four_outputs, dim=1),
                                             F.softmax(Variable(third_outputs), dim=1))
                kl_loss4 = args.criterion_kl(F.log_softmax(four_outputs, dim=1),
                                             F.softmax(Variable(five_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 + kl_loss3 + kl_loss4) / 4  # 總損失為交叉熵損失與 KL 散度損失之和

                loss.backward()  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.four_model.parameters(), args.clip)  # 梯度裁剪
                four_optimizer.step()  # 更新其他模型參數

                # train five_model 訓練其他模型
                ce_loss = args.criterion_ce(five_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(five_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(five_outputs, dim=1),
                                             F.softmax(Variable(other_outputs), dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(five_outputs, dim=1),
                                             F.softmax(Variable(third_outputs), dim=1))
                kl_loss4 = args.criterion_kl(F.log_softmax(five_outputs, dim=1),
                                             F.softmax(Variable(four_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 + kl_loss3 + kl_loss4) / 4  # 總損失為交叉熵損失與 KL 散度損失之和

                loss.backward()  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.five_model.parameters(), args.clip)  # 梯度裁剪
                five_optimizer.step()  # 更新其他模型參數

        acc_train = 100.0 * correct / count
        loss_train = running_loss / len(self.trainloader)

        return y_true_train, y_pred_train, acc_train, loss_train

    def local_train4(self):
        # 將模型移到指定的設備上
        self.local_model = self.local_model.to(args.device)
        self.other_model = self.other_model.to(args.device)
        self.third_model = self.third_model.to(args.device)
        self.four_model = self.four_model.to(args.device)

        # 定義優化器
        local_optimizer = optim.SGD(self.local_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        other_optimizer = optim.SGD(self.other_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        third_optimizer = optim.SGD(self.third_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        four_optimizer = optim.SGD(self.four_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # 設定模型為訓練模式
        self.local_model.train()
        self.other_model.train()
        self.third_model.train()
        self.four_model.train()

        y_true_train = []
        y_pred_train = []


        for epoch in range(args.local_epochs):
            running_loss = 0.0
            correct = 0
            count = 0
            # 讀取訓練資料
            for (data, labels) in self.trainloader:
                torch.autograd.set_detect_anomaly(True)
                # 將資料和標籤轉為變量並移到指定的設備上
                data, labels = Variable(data), Variable(labels)
                data, labels = data.to(args.device), labels.to(args.device)
                # 清空優化器的梯度
                local_optimizer.zero_grad()
                other_optimizer.zero_grad()
                third_optimizer.zero_grad()
                four_optimizer.zero_grad()
                # 進行前向傳播
                local_outputs = self.local_model(data)
                other_outputs = self.other_model(data)
                third_outputs = self.third_model(data)
                four_outputs = self.four_model(data)

                labels = labels.to(torch.long)
                labels = labels.squeeze()
                # print("Modified target labels /shape:", labels.shape)
                # labels = labels.long()

                # print("Labels Shape:", labels.shape)
                if len(labels.shape) > 1:
                    labels = torch.argmax(labels, dim=1).long()


                # train local_model訓練本地模型
                ce_loss = args.criterion_ce(local_outputs, labels)  # 計算交叉熵損失(交叉熵損失（用於度量模型對目標標籤的預測能力）)
                kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                            F.softmax(Variable(other_outputs),
                                                      dim=1))  # 計算 KL 散度損失(用於度量兩個模型輸出之間的相似性)
                kl_loss2 = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                            F.softmax(Variable(third_outputs),
                                                      dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                             F.softmax(Variable(four_outputs),
                                                       dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 + kl_loss3)/3  # 總損失為交叉熵損失與 KL 散度損失之和
                running_loss = running_loss+loss.item()
                predicted = torch.argmax(local_outputs, dim=1)  # 預測類別
                correct += (predicted == labels).sum().item()  # 計算正確預測的數量
                count += len(labels)  # 累計資料數量

                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())


                loss.backward(retain_graph=True)  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)  # 梯度裁剪
                local_optimizer.step()  # 更新本地模型參數
                # train other_model 訓練其他模型
                ce_loss = args.criterion_ce(other_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                            F.softmax(Variable(third_outputs), dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                             F.softmax(Variable(four_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 +kl_loss3)/3 # 總損失為交叉熵損失與 KL 散度損失之和
                loss.backward(retain_graph=True)  # 反向傳播計算梯度                running_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.other_model.parameters(), args.clip)  # 梯度裁剪
                other_optimizer.step()  # 更新其他模型參數

                # train third_model 訓練其他模型
                ce_loss = args.criterion_ce(third_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                             F.softmax(Variable(other_outputs), dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                             F.softmax(Variable(four_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 + kl_loss3)/3  # 總損失為交叉熵損失與 KL 散度損失之和

                loss.backward()  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.third_model.parameters(), args.clip)  # 梯度裁剪
                third_optimizer.step()  # 更新其他模型參數

                # train four_model 訓練其他模型
                ce_loss = args.criterion_ce(four_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(four_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(four_outputs, dim=1),
                                             F.softmax(Variable(other_outputs), dim=1))
                kl_loss3 = args.criterion_kl(F.log_softmax(four_outputs, dim=1),
                                             F.softmax(Variable(third_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2 + kl_loss3) / 3  # 總損失為交叉熵損失與 KL 散度損失之和

                loss.backward()  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.four_model.parameters(), args.clip)  # 梯度裁剪
                four_optimizer.step()  # 更新其他模型參數

        acc_train = 100.0 * correct / count
        loss_train = running_loss / len(self.trainloader)

        return y_true_train, y_pred_train, acc_train, loss_train

    def local_train3(self):
        # 將模型移到指定的設備上
        self.local_model = self.local_model.to(args.device)
        self.other_model = self.other_model.to(args.device)
        self.third_model = self.third_model.to(args.device)

        # 定義優化器
        local_optimizer = optim.SGD(self.local_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        other_optimizer = optim.SGD(self.other_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        third_optimizer = optim.SGD(self.third_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # 設定模型為訓練模式
        self.local_model.train()
        self.other_model.train()
        self.third_model.train()

        y_true_train = []
        y_pred_train = []
        # 進行本地訓練迭代
        for epoch in range(args.local_epochs):
            running_loss = 0.0
            correct = 0
            count = 0
            # 讀取訓練資料
            for (data, labels) in self.trainloader:
                torch.autograd.set_detect_anomaly(True)
                # 將資料和標籤轉為變量並移到指定的設備上
                data, labels = Variable(data), Variable(labels)
                data, labels = data.to(args.device), labels.to(args.device)
                # 清空優化器的梯度
                local_optimizer.zero_grad()
                other_optimizer.zero_grad()
                third_optimizer.zero_grad()
                # 進行前向傳播
                local_outputs = self.local_model(data)
                other_outputs = self.other_model(data)
                third_outputs = self.third_model(data)

                labels = labels.to(torch.long)
                labels = labels.squeeze()
                # print("Modified target labels /shape:", labels.shape)
                # labels = labels.long()

                # print("Labels Shape:", labels.shape)
                if len(labels.shape) > 1:
                    labels = torch.argmax(labels, dim=1).long()


                # train local_model訓練本地模型
                ce_loss = args.criterion_ce(local_outputs, labels)  # 計算交叉熵損失(交叉熵損失（用於度量模型對目標標籤的預測能力）)
                kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                            F.softmax(Variable(other_outputs),
                                                      dim=1))  # 計算 KL 散度損失(用於度量兩個模型輸出之間的相似性)
                kl_loss2 = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                            F.softmax(Variable(third_outputs),
                                                      dim=1))
                loss = ce_loss + (kl_loss + kl_loss2)/2  # 總損失為交叉熵損失與 KL 散度損失之和
                running_loss = running_loss+loss.item()
                predicted = torch.argmax(local_outputs, dim=1)  # 預測類別
                correct += (predicted == labels).sum().item()  # 計算正確預測的數量
                count += len(labels)  # 累計資料數量
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())


                loss.backward(retain_graph=True)  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)  # 梯度裁剪
                local_optimizer.step()  # 更新本地模型參數
                # train other_model 訓練其他模型
                ce_loss = args.criterion_ce(other_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                            F.softmax(Variable(third_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2)/2 # 總損失為交叉熵損失與 KL 散度損失之和
                loss.backward(retain_graph=True)  # 反向傳播計算梯度                running_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.other_model.parameters(), args.clip)  # 梯度裁剪
                other_optimizer.step()  # 更新其他模型參數

                # train third_model 訓練其他模型
                ce_loss = args.criterion_ce(third_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失
                kl_loss2 = args.criterion_kl(F.log_softmax(third_outputs, dim=1),
                                             F.softmax(Variable(other_outputs), dim=1))
                loss = ce_loss + (kl_loss + kl_loss2)/2  # 總損失為交叉熵損失與 KL 散度損失之和

                loss.backward()  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.third_model.parameters(), args.clip)  # 梯度裁剪
                third_optimizer.step()  # 更新其他模型參數

        acc_train = 100.0 * correct / count
        loss_train = running_loss / len(self.trainloader)

        return y_true_train, y_pred_train, acc_train, loss_train

    def local_train2(self):
        # 將模型移到指定的設備上
        self.local_model = self.local_model.to(args.device)
        self.other_model = self.other_model.to(args.device)
        # self.third_model = self.third_model.to(args.device)

        # 定義優化器
        local_optimizer = optim.SGD(self.local_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        other_optimizer = optim.SGD(self.other_model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # third_optimizer = optim.SGD(self.third_model.parameters(), lr=args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        # 設定模型為訓練模式
        self.local_model.train()
        self.other_model.train()

        y_true_train = []
        y_pred_train = []
        # self.third_model.train()
        # 進行本地訓練迭代
        for epoch in range(args.local_epochs):
            running_loss = 0.0
            correct = 0
            count = 0
            # 讀取訓練資料
            for (data, labels) in self.trainloader:
                torch.autograd.set_detect_anomaly(True)
                # 將資料和標籤轉為變量並移到指定的設備上
                data, labels = Variable(data), Variable(labels)
                data, labels = data.to(args.device), labels.to(args.device)
                # 清空優化器的梯度
                local_optimizer.zero_grad()
                other_optimizer.zero_grad()
                # third_optimizer.zero_grad()
                # 進行前向傳播
                local_outputs = self.local_model(data)
                other_outputs = self.other_model(data)
                # third_outputs = self.third_model(data)

                labels = labels.to(torch.long)
                labels = labels.squeeze()
                # print("Modified target labels /shape:", labels.shape)
                # labels = labels.long()

                # print("Labels Shape:", labels.shape)
                if len(labels.shape) > 1:
                    labels = torch.argmax(labels, dim=1).long()


                # train local_model訓練本地模型
                ce_loss = args.criterion_ce(local_outputs, labels)  # 計算交叉熵損失(交叉熵損失（用於度量模型對目標標籤的預測能力）)
                kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim=1),
                                            F.softmax(Variable(other_outputs),
                                                      dim=1))  # 計算 KL 散度損失(用於度量兩個模型輸出之間的相似性)

                loss = ce_loss + kl_loss  # 總損失為交叉熵損失與 KL 散度損失之和
                running_loss = running_loss+loss.item()
                predicted = torch.argmax(local_outputs, dim=1)  # 預測類別
                correct += (predicted == labels).sum().item()  # 計算正確預測的數量
                count += len(labels)  # 累計資料數量

                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())


                loss.backward(retain_graph=True)  # 反向傳播計算梯度
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)  # 梯度裁剪
                local_optimizer.step()  # 更新本地模型參數
                # train other_model 訓練其他模型
                ce_loss = args.criterion_ce(other_outputs, labels)  # 計算交叉熵損失
                kl_loss = args.criterion_kl(F.log_softmax(other_outputs, dim=1),
                                            F.softmax(Variable(local_outputs), dim=1))  # 計算 KL 散度損失

                loss = ce_loss + kl_loss # 總損失為交叉熵損失與 KL 散度損失之和
                loss.backward(retain_graph=True)  # 反向傳播計算梯度                running_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.other_model.parameters(), args.clip)  # 梯度裁剪
                other_optimizer.step()  # 更新其他模型參數

                # train third_model 訓練其他模型

        acc_train = 100.0 * correct / count
        loss_train = running_loss / len(self.trainloader)

        return y_true_train, y_pred_train, acc_train, loss_train



    # 驗證函數，評估本地模型在驗證數據集上的性能
    def validate(self):
        y_true_valid, y_pred_valid, acc_valid_tmp, loss_valid_tmp = test(self.local_model, args.criterion_ce,
                                                                         self.valloader)
        return y_true_valid, y_pred_valid, acc_valid_tmp, loss_valid_tmp

    # 模型替換函數，用於根據在驗證集上的性能決定是否替換本地模型
    def model_replacement(self):
        _, _, _, loss_local = test(self.local_model, args.criterion_ce, self.valloader)
        _, _, _, loss_other = test(self.other_model, args.criterion_ce, self.valloader)
        # 如果其他模型的損失小於本地模型的損失，則替換本地模型
        if loss_other < loss_local:
            self.local_model_id = self.other_model_id


# 定義訓練函數，該函數接受模型、損失函數、訓練數據加載器和訓練迭代次數作為輸入
def train(model, criterion, trainloader, epochs):
    # 初始化優化器，使用隨機梯度下降法(SGD)進行優化
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model.train()  # 設定模型為訓練模式
    # 迭代指定次數的訓練過程
    for epoch in range(epochs):  # 讀取訓練數據
        running_loss = 0.0
        correct = 0
        count = 0
        for (data, labels) in trainloader:
            # 將數據和標籤轉為變量並移到指定的設備上
            data, labels = Variable(data), Variable(labels)
            data, labels = data.to(args.device), labels.to(args.device)

            labels = labels.to(torch.long)
            labels = labels.squeeze()
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1).long()


            optimizer.zero_grad()  # 清空優化器的梯度
            outputs = model(data)  # 進行前向傳播
            # 計算損失
            loss = criterion(outputs, labels)  # 計算損失
            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)  # 預測類別
            # 計算正確預測的數量
            correct += (predicted == labels).sum().item()
            count += len(labels)
            loss.backward()  # 反向傳播計算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 梯度裁剪
            optimizer.step()  # 更新模型參數


    return 100.0 * correct / count, running_loss / len(trainloader)  # 返回訓練準確率和平均損失


# 定義測試函數，該函數接受模型、損失函數和測試數據加載器作為輸入
def test(model, criterion, testloader):
    model.eval()  # 設定模型為評估模式
    running_loss = 0.0
    correct = 0
    count = 0
    y_true = []
    y_pred = []
    for (data, labels) in testloader:
        data, labels = data.to(args.device), labels.to(args.device)
        outputs = model(data)

        import torch.nn.functional as F
        outputs = F.log_softmax(model(data), dim=1)
        labels = labels.view(-1).long()

        running_loss += criterion(outputs, labels).item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        count += len(labels)
    # 計算準確率和平均損失
        # Collect true and predicted labels
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    # 計算準確率和平均損失
    accuracy = 100.0 * correct / count
    loss = running_loss / len(testloader)

    return y_true, y_pred, accuracy, loss

def test3(model, criterion, testloader):
    model.eval()  # 設定模型為評估模式
    running_loss = 0.0
    correct = 0
    count = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for (data, labels) in testloader:
            data, labels = data.to(args.device), labels.to(args.device)
            outputs = model(data)

            import torch.nn.functional as F
            outputs = F.log_softmax(model(data), dim=1)
            labels = labels.view(-1).long()

            running_loss += criterion(outputs, labels).item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            count += len(labels)

            # Collect true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 計算準確率和平均損失
    accuracy = 100.0 * correct / count
    loss = running_loss / len(testloader)

    # 計算 F1, Recall 和 Precision
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return y_true, y_pred, accuracy, loss, precision, recall, f1

class Early_Stopping():
    def __init__(self, partience):
        self.step = 0
        self.loss = float('inf')
        self.partience = partience

    def validate(self, loss):
        if self.loss < loss:
            self.step += 1
            if self.step > self.partience:
                return True
        else:
            self.step = 0
            self.loss = loss

        return False




server = Server(global_trainloader)
workers = server.create_worker(federated_trainset, federated_valset, federated_testset, client_model)
acc_train = []
loss_train = []
acc_valid = []
loss_valid = []
f1_train=[]
recall_train=[]
precision_train=[]
f1_valid=[]
recall_valid=[]
precision_valid=[]

early_stopping = Early_Stopping(args.partience)
# 記錄訓練開始時間
start_time = time.time()


# 初始化變量
stable_epochs = 0
patience = 4  # 耐心參數
stable_threshold = 0.1  # 滑動平均變化閾值

# 用於存儲準確率的列表
acc_valid_history = []

# 訓練方法替換階段標記
train_stage = 5
from sklearn.metrics import f1_score, recall_score, precision_score

for epoch in range(args.global_epochs):
    if epoch in args.turn_of_cluster_num:
        idx = args.turn_of_cluster_num.index(epoch)
        args.cluster_num = args.cluster_list[idx]
    sample_worker = server.sample_worker(workers)  # 隨機抽樣工作者
    server.collect_model(sample_worker)  # 收集工作者的模型
    server.clustering(sample_worker)  # 對工作者進行聚類
    server.decide_other_model(sample_worker)  # 為工作者決定其他模型
    server.send_model(sample_worker)  # 向工作者發送模型
    # 初始化各工作者的平均訓練和驗證準確率和損失
    acc_train_avg = 0.0
    loss_train_avg = 0.0
    acc_valid_avg = 0.0
    loss_valid_avg = 0.0

    f1_train_avg, recall_train_avg, precision_train_avg = 0.0, 0.0, 0.0
    f1_valid_avg, recall_valid_avg, precision_valid_avg = 0.0, 0.0, 0.0
    # 對每個抽樣的工作者進行本地訓練和驗證
    print("train_stage", train_stage)
    for worker in sample_worker:
        if train_stage == 5:
            # acc_train_tmp, loss_train_tmp = worker.local_train5()
            y_true_train, y_pred_train, acc_train_tmp, loss_train_tmp = worker.local_train5()
            y_true_valid, y_pred_valid, acc_valid_tmp, loss_valid_tmp = worker.validate()

            # Calculate the average training metrics
            acc_train_avg += acc_train_tmp / len(sample_worker)
            loss_train_avg += loss_train_tmp / len(sample_worker)
            acc_valid_avg += acc_valid_tmp / len(sample_worker)
            loss_valid_avg += loss_valid_tmp / len(sample_worker)

            # Calculate F1, Recall, Precision with zero_division parameter
            f1_train_avg += f1_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(sample_worker)
            recall_train_avg += recall_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)
            precision_train_avg += precision_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)

            f1_valid_avg += f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(sample_worker)
            recall_valid_avg += recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)
            precision_valid_avg += precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)
        elif train_stage == 4:
            y_true_train, y_pred_train, acc_train_tmp, loss_train_tmp = worker.local_train5()
            y_true_valid, y_pred_valid, acc_valid_tmp, loss_valid_tmp = worker.validate()

            # Calculate the average training metrics
            acc_train_avg += acc_train_tmp / len(sample_worker)
            loss_train_avg += loss_train_tmp / len(sample_worker)
            acc_valid_avg += acc_valid_tmp / len(sample_worker)
            loss_valid_avg += loss_valid_tmp / len(sample_worker)

            # Calculate F1, Recall, Precision with zero_division parameter
            f1_train_avg += f1_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(sample_worker)
            recall_train_avg += recall_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)
            precision_train_avg += precision_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)

            f1_valid_avg += f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(sample_worker)
            recall_valid_avg += recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)
            precision_valid_avg += precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)
        elif train_stage == 3:
            y_true_train, y_pred_train, acc_train_tmp, loss_train_tmp = worker.local_train5()
            y_true_valid, y_pred_valid, acc_valid_tmp, loss_valid_tmp = worker.validate()

            # Calculate the average training metrics
            acc_train_avg += acc_train_tmp / len(sample_worker)
            loss_train_avg += loss_train_tmp / len(sample_worker)
            acc_valid_avg += acc_valid_tmp / len(sample_worker)
            loss_valid_avg += loss_valid_tmp / len(sample_worker)

            # Calculate F1, Recall, Precision with zero_division parameter
            f1_train_avg += f1_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(sample_worker)
            recall_train_avg += recall_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)
            precision_train_avg += precision_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)

            f1_valid_avg += f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(sample_worker)
            recall_valid_avg += recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)
            precision_valid_avg += precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)
        elif train_stage == 2:
            y_true_train, y_pred_train, acc_train_tmp, loss_train_tmp = worker.local_train5()
            y_true_valid, y_pred_valid, acc_valid_tmp, loss_valid_tmp = worker.validate()

            # Calculate the average training metrics
            acc_train_avg += acc_train_tmp / len(sample_worker)
            loss_train_avg += loss_train_tmp / len(sample_worker)
            acc_valid_avg += acc_valid_tmp / len(sample_worker)
            loss_valid_avg += loss_valid_tmp / len(sample_worker)

            # Calculate F1, Recall, Precision with zero_division parameter
            f1_train_avg += f1_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(sample_worker)
            recall_train_avg += recall_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)
            precision_train_avg += precision_score(y_true_train, y_pred_train, average='macro', zero_division=0) / len(
                sample_worker)

            f1_valid_avg += f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(sample_worker)
            recall_valid_avg += recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)
            precision_valid_avg += precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0) / len(
                sample_worker)



    # 將當前準確率加入歷史記錄
    acc_valid_history.append(acc_valid_avg)



    # 如果需要進行模型替換
    if epoch in args.turn_of_replacement_model:
        for worker in sample_worker:
            worker.model_replacement()
    if train_stage == 5:
        server.aggregate_model5(sample_worker)  # 聚合工作者的模型
    elif train_stage == 4:
        server.aggregate_model4(sample_worker)  # 聚合工作者的模型
    elif train_stage == 3:
        server.aggregate_model3(sample_worker)  # 聚合工作者的模型
    elif train_stage == 2:
        server.aggregate_model2(sample_worker)  # 聚合工作者的模型
    #檢查準確率是否趨於穩定

    # 檢查準確率是否趨於穩定
    if len(acc_valid_history) > patience:
        recent_avg = sum(acc_valid_history[-patience:]) / patience
        previous_avg = sum(acc_valid_history[-(patience * 2):-patience]) / patience
        if abs(recent_avg - previous_avg) < stable_threshold:
            stable_epochs += 1
            if stable_epochs >= patience:
                # 轉換到更輕量級的訓練方法
                if train_stage == 5:
                    for worker in sample_worker:
                        worker.local_train5 = worker.local_train4
                    train_stage = 4
                elif train_stage == 4:
                    for worker in sample_worker:
                        worker.local_train4 = worker.local_train3
                    train_stage = 3
                elif train_stage == 3:
                    for worker in sample_worker:
                        worker.local_train3 = worker.local_train2
                    train_stage = 2
                stable_epochs = 0  # 重置穩定計數器
        else:
            stable_epochs = 0



    server.return_model(sample_worker)  # 返回工作者的模型
    print('Epoch {}  loss: {:.4f}  accuracy: {:.4f}  F1: {:.4f}  Recall: {:.4f}  Precision: {:.4f}'.format(
        epoch + 1, loss_valid_avg, acc_valid_avg, f1_valid_avg, recall_valid_avg, precision_valid_avg))


    # 更新訓練和驗證的準確率和損失列表
    acc_train.append(acc_train_avg)
    loss_train.append(loss_train_avg)
    acc_valid.append(acc_valid_avg)
    loss_valid.append(loss_valid_avg)
    f1_train.append(f1_train_avg)
    recall_train.append(recall_train_avg)
    precision_train.append(precision_train_avg)
    f1_valid.append(f1_valid_avg)
    recall_valid.append(recall_valid_avg)
    precision_valid.append(precision_valid_avg)


    if early_stopping.validate(loss_valid_avg):
        print('Early Stop')
        break


min_length = min(len(acc_train), len(loss_train), len(acc_valid), len(loss_valid))
acc_train = acc_train[:min_length]
loss_train = loss_train[:min_length]
acc_valid = acc_valid[:min_length]
loss_valid = loss_valid[:min_length]
f1_valid = f1_valid[:min_length]
recall_valid = recall_valid[:min_length]
precision_valid =precision_valid[:min_length]


results_df = pd.DataFrame({
    'Epoch': range(1, min_length + 1),
    'Training Loss': loss_valid,
    'Test Accuracy': acc_valid,
    "F1": f1_valid ,
    "Recall": recall_valid,
    "Precision": precision_valid,    # 'Parameter': total_params1 + total_params2
})
from pathlib import Path

# Ensure the directory exists
save_path = Path('/home/icmnlab/LEE/dataset/train/epoch_result/fedme_me')
save_path.mkdir(parents=True, exist_ok=True)

results_df.to_csv(save_path / 'proposedHRLR_5M_01_good_epoch.csv', index=False)
print(f'Epoch results saved to {save_path / "proposedHRLR_5M_01_good_epoch.csv"}')

end_time = time.time()
execution_time = end_time - start_time

print(f"Total Training Time: {execution_time:.2f} seconds")

acc_test = []
loss_test = []

start = time.time()

# 對每個工作者進行測試
acc_test, loss_test, precision_test, recall_test, f1_test = [], [], [], [], []

for i, worker in enumerate(workers):
    worker.local_model = worker.local_model.to(args.device)  # 將本地模型轉移到設備（例如 GPU）
    y_true, y_pred, acc_tmp, loss_tmp, precision_tmp, recall_tmp, f1_tmp = test3(worker.local_model, args.criterion_ce, worker.testloader)  # 測試本地模型並獲取準確率和損失
    acc_test.append(acc_tmp)  # 將測試準確率和損失添加到列表中
    loss_test.append(loss_tmp)  # 將測試損失添加到列表中
    precision_test.append(precision_tmp)  # 將測試 precision 添加到列表中
    recall_test.append(recall_tmp)  # 將測試 recall 添加到列表中
    f1_test.append(f1_tmp)  # 將測試 F1 score 添加到列表中
    print('Worker{} accuracy:{}  loss:{}  precision:{}  recall:{}  F1:{}'.format(i + 1, acc_tmp, loss_tmp, precision_tmp, recall_tmp, f1_tmp))
    worker.local_model = worker.local_model.to('cpu')  # 將本地模型轉回 CPU


end = time.time()
# 計算並輸出測試集的平均準確率和損失
acc_test_avg = sum(acc_test) / len(acc_test)
loss_test_avg = sum(loss_test) / len(loss_test)
print('Test  loss:{}  accuracy:{}'.format(loss_test_avg, acc_test_avg))



acc_tune_test = []
loss_tune_test = []
acc_tune_valid = []
loss_tune_valid = []

pltacc = []
pltloss = []
pltworker = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
start = time.time()

acc_tune_valid, loss_tune_valid, precision_tune_valid, recall_tune_valid, f1_tune_valid = [], [], [], [], []
acc_tune_test, loss_tune_test, precision_tune_test, recall_tune_test, f1_tune_test = [], [], [], [], []
pltacc, pltloss = [], []


for i, worker in enumerate(workers):
    worker.local_model = worker.local_model.to(args.device)
    _, _ = train(worker.local_model, args.criterion_ce, worker.trainloader, args.local_epochs)  # Train the local model
    # Validate tuned model on validation set
    y_true_val, y_pred_val, acc_tmp_val, loss_tmp_val, precision_val, recall_val, f1_val = test3(worker.local_model, args.criterion_ce, worker.valloader)
    acc_tune_valid.append(acc_tmp_val)
    loss_tune_valid.append(loss_tmp_val)
    precision_tune_valid.append(precision_val)
    recall_tune_valid.append(recall_val)
    f1_tune_valid.append(f1_val)
    print('Worker{} Valid accuracy:{}  loss:{}  precision:{}  recall:{}  F1:{}'.format(i + 1, acc_tmp_val, loss_tmp_val, precision_val, recall_val, f1_val))

    # Test tuned model on test set
    y_true_test, y_pred_test, acc_tmp_test, loss_tmp_test, precision_test, recall_test, f1_test = test3(worker.local_model, args.criterion_ce, worker.testloader)
    acc_tune_test.append(acc_tmp_test)
    loss_tune_test.append(loss_tmp_test)
    precision_tune_test.append(precision_test)
    recall_tune_test.append(recall_test)
    f1_tune_test.append(f1_test)
    print('Worker{} Test accuracy:{}  loss:{}  precision:{}  recall:{}  F1:{}'.format(i + 1, acc_tmp_test, loss_tmp_test, precision_test, recall_test, f1_test))

    worker.local_model = worker.local_model.to('cpu')

results_df = pd.DataFrame({
    'workers': range(1, args.worker_num + 1),
    'valid Loss': loss_tune_valid,
    'valid Accuracy': acc_tune_valid,
    'test Loss': loss_tune_test,
    'test Accuracy': acc_tune_test,
    "f1": f1_tune_valid,
    "precision":precision_tune_valid,
    "recall":recall_tune_valid
    # 'Parameter': total_params1 + total_params2
})
from pathlib import Path

# Ensure the directory exists
save_path = Path('/home/icmnlab/LEE/dataset/train/epoch_result/fedme_me')
save_path.mkdir(parents=True, exist_ok=True)

results_df.to_csv(save_path / 'proposedHRLR_5M_01_good_worker.csv', index=False)
print(f'Epoch results saved to {save_path / "proposedHRLR_5M_01_good_worker.csv"}')

epochs = list(range(1, len(acc_train) + 1))
plt.figure(figsize=(10, 5))
# Plot validation accuracy
plt.plot(epochs, acc_train, label='Train Accuracy', color='blue', marker='o')
# Plot test accuracy
plt.plot(epochs, acc_valid, label='Validation Accuracy', color='red', marker='x')
# Adding titles and labels
plt.title('Train and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# Adding a legend
plt.legend()
# plt.show()
plt.figure(figsize=(10, 5))  # 1 row, 2 columns, 2nd subplot
plt.plot(epochs, loss_train, label='train Loss', color='green', marker='o')
plt.plot(epochs, loss_valid, label='Validation Loss', color='purple', marker='x')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



end = time.time()

acc_valid_avg = sum(acc_tune_valid) / len(acc_tune_valid)
loss_valid_avg = sum(loss_tune_valid) / len(loss_tune_valid)
print('Validation(tune)  loss:{}  accuracy:{}'.format(loss_valid_avg, acc_valid_avg))
acc_test_avg = sum(acc_tune_test) / len(acc_tune_test)
loss_test_avg = sum(loss_tune_test) / len(loss_tune_test)
print('Test(tune)  loss:{}  accuracy:{}'.format(loss_test_avg, acc_test_avg))

end_time = datetime.datetime.now()
# print("time used", (end_time - start_time))