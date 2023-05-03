import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if args.dataset == 'cub200':
            self.K =3000

        else:
            self.K = 30000
        # self.num_features = 512
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200','manyshotcub']:
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        self.register_buffer("queue", torch.randn(self.num_features, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr1", torch.zeros(1, dtype=torch.long))
        self.register_buffer("lab_que", torch.randint(0, self.args.num_classes, [self.K, ]))

        self.centroid = torch.randn(self.args.num_classes, self.num_features)
        self.cov = torch.randn(self.args.num_classes, self.num_features, self.num_features)


    def _dequeue_and_enqueue(self, high_feature, train_label):
        # ptr 入队指针，queue 特征存储队列，lab_que 标签队列
        batch_size = high_feature.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            # 如果超出队列长度，需要从头开始存储
            remaining_size = self.K - ptr
            self.queue[:, ptr:] = high_feature[:remaining_size].T
            self.lab_que[ptr:] = train_label[:remaining_size]
            self.queue[:, :batch_size - remaining_size] = high_feature[remaining_size:].T
            self.lab_que[:batch_size - remaining_size] = train_label[remaining_size:]
            ptr = batch_size - remaining_size
        else:
            # 直接存储
            self.queue[:, ptr:ptr + batch_size] = high_feature.T
            self.lab_que[ptr:ptr + batch_size] = train_label
            ptr += batch_size
        self.queue_ptr[0] = ptr % self.K  # 更新指针位置

    def forward_metric(self, x):
        x = self.encode(x)
        temp = x
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return temp,x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            temp,input = self.forward_metric(input)
            return temp,input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def encode_fc(self,temp):
        if 'cos' in self.mode:
            temp = F.linear(F.normalize(temp, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))  # 16 * 200
            # temp = F.linear(F.normalize(temp, p=2, dim=-1), self.fc.weight)  # 16 * 200
            temp = self.args.temperature * temp

        elif 'dot' in self.mode:
            temp = self.fc(temp)
            temp = self.args.temperature * temp
        return temp

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            #print(class_index)
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            #print(proto)
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def cov_loss(self,x,index):

        cov = self.cov[index, :, :].cuda()
        # x_normalized = F.normalize(x, p=2, dim=-1)
        temp = []
        for i in range(x.shape[0]):
            # temp.append(F.linear(x[i], F.normalize(cov[i], p=2, dim=-1)))
            temp.append(F.linear(x[i],cov[i] ))
        noise = torch.stack(temp)/self.num_features
        rand_matrix = torch.rand(size=(1, self.num_features)) + 1e-6
        rand_matrix = rand_matrix.repeat(noise.shape[0], 1).cuda()
        temp =  x + noise * rand_matrix
        return  temp


    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

