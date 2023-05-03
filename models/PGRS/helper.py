# import new Network name here and add in model_class args
from .Network import *
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
import torch.nn as nn
from sklearn.mixture import GaussianMixture

criterion = torch.nn.CrossEntropyLoss().cuda()
softmax_func = nn.Softmax(dim = 1).cuda()
logsoftmax_func = nn.LogSoftmax(dim=1).cuda()
dropout = nn.Dropout(p=0.5)

def Robust_loss(logits, label,alph,TH):
    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = logits.to(device)
    label = label.to(device)

    logsoftmax_output = torch.nn.functional.log_softmax(logits, dim=1)

    # Get the confidence scores
    softmax = torch.nn.functional.softmax(logsoftmax_output, dim=1)
    max_score, max_label = softmax.max(dim=1)

    batch_size = logits.shape[0]

    weights1 = torch.zeros(batch_size)
    weights2 = torch.zeros(batch_size)
    weights3 = torch.zeros(batch_size)

    for i in range(batch_size):
        if max_label[i] == label[i]:
            weights1[i] = 2 - max_score[i]
        else:
            if max_score[i] < TH:
               weights2[i] = 2 - max_score[i]
            else:
              weights3[i] = 1 - max_score[i]


    weights = (1 - alph) * weights1 + (1-alph) * weights2 + (1 + alph) * weights3
    weights  = weights.cuda()
    weighted_losses = -1 * logsoftmax_output[torch.arange(batch_size), label] * weights
    total_loss = torch.sum(weighted_losses) / batch_size
    return total_loss


def cos_f(a, b):
    if len(a.shape)==1 and len(b.shape)==1:
        a = a.unsqueeze(dim = 0)
        b = b.unsqueeze(dim = 0)
    cos = nn.CosineSimilarity()
    similiarity = cos(a, b)
    return similiarity




def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    """
        Trains a deep learning model on a training dataset.

        Args:
            model: the deep learning model to be trained.
            trainloader: the data loader for the training dataset.
            optimizer: the optimizer to be used for updating the model's parameters.
            scheduler: the learning rate scheduler to be used.
            epoch: the current epoch number.
            args: additional arguments for the training process.

        Returns:
            The total loss and accuracy of the model on the training dataset.
        """

    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    args.epoch = epoch

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        high_feture, logits = model(data)
        train_label_np = train_label.cpu().numpy()

        label_new = (logits[:, args.base_class:].argmax(dim=1) + args.base_class).cuda()

        # 协方差加噪
        cov_feture1 = model.module.cov_loss(high_feture, train_label_np)
        cov_logits1 = model.module.encode_fc(cov_feture1)
        cov_loss = Robust_loss(cov_logits1, train_label, epoch / args.epochs_base,args.Th)

        noise = torch.randn_like(high_feture).cuda() * args.std

        high_feture_new = dropout(high_feture)

        # num_samples_to_add = int(epoch/ args.epochs_base)
        num_samples_to_add = data.shape[0]
        increment_data = high_feture[:num_samples_to_add] + noise[:num_samples_to_add]
        increment_label = label_new[:num_samples_to_add]

        increment_data = torch.cat((high_feture_new,increment_data),dim=0)
        increment_label = torch.cat((label_new,increment_label),dim = 0)

        logits_new = model.module.encode_fc(increment_data)
        Robust_loss1 = F.cross_entropy(logits_new, increment_label)


        loss1 = Robust_loss(logits_new, increment_label, epoch / args.epochs_base,args.Th)

        loss = F.cross_entropy(logits[:,:args.base_class], train_label)

        total_loss =  loss +  cov_loss + 1*( loss1 +Robust_loss1 )

        acc = count_acc(logits, train_label)

        # 输出显示
        lrc = optimizer.state_dict()['param_groups'][0]['lr']
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f} '.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        model.module._dequeue_and_enqueue(high_feture, train_label)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()

    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []

    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)
            label_np = label.cpu().numpy()
            # 通过协方差向量进一步精确原型
            # embedding = model.module.cov_loss(embedding, label_np)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])


    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            _, logits3 = model(data)
            logits3 = logits3[:, :test_class]
            loss = F.cross_entropy(logits3, test_label)
            acc = count_acc(logits3, test_label)
            top5acc = count_acc_topk(logits3, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt = torch.cat([lgt, logits3.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f} '.format(epoch, vl, va, va5))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)

        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)


    return vl, va


