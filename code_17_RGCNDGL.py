# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 08:03:20 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import dgl
from dgl.nn.pytorch.conv import RelGraphConv

import re
import numpy as np
import pandas as pd

from code_15_BERT_PROPN import (device, df_test, df_train_val, getmodel)

import spacy
import pickle
import collections

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn import metrics

'''加载预处理文件'''
base_dir = 'output/'

# 训练数据
# tokens_NoPUNC     2454,266
# offsets_NoPUNC    2454,3
# PROPN_bert        2454,1,3,768
# bert_forNoPUNC    2454,1,266,768
offsets_NoPUNC = pickle.load(open(base_dir + 'offsets_NoPUNC.pkl', "rb"))
tokens_NoPUNC = pickle.load(open(base_dir + 'tokens_NoPUNC_padding.pkl', "rb"))  # tokens of every sentence without padding
bert_forNoPUNC = pickle.load(open(base_dir + 'bert_outputs_forNoPUNC.pkl', "rb"))  # list of outputs of bert for every sentence
PROPN_bert = pickle.load(open(base_dir + 'bert_outputs_forPROPN.pkl', "rb"))

# 测试数据
test_offsets_NoPUNC = pickle.load(open(base_dir + 'test_offsets_NoPUNC.pkl', "rb"))
test_tokens_NoPUNC = pickle.load(
    open(base_dir + 'test_tokens_NoPUNC_padding.pkl', "rb"))  # tokens of every sentence without padding
test_bert_forNoPUNC = pickle.load(
    open(base_dir + 'test_bert_outputs_forNoPUNC.pkl', "rb"))  # list of outputs of bert for every sentence
test_PROPN_bert = pickle.load(open(base_dir + 'test_bert_outputs_forPROPN.pkl', "rb"))


tokenizer, _ = getmodel()  # 加载BERT分词工具
# parser = spacy.load('en')  # 加载SpaCy模型  'en_core_web_sm')#en_core_web_lg
parser = spacy.load("en_core_web_sm")

########################################################################

# 生成图结构数据
def getGraphsData(tokens_NoPUNC, offsets_NoPUNC, PROPN_bert, bert_forNoPUNC):
    # 2454
    assert len(tokens_NoPUNC) == len(offsets_NoPUNC) == len(PROPN_bert) == len(bert_forNoPUNC)
    # 266=269-3. cod_16: max_len = 269  # 设置处理文本的最大长度
    assert len(tokens_NoPUNC[0]) == len(bert_forNoPUNC[0][0])
    assert len(offsets_NoPUNC[0]) == len(PROPN_bert[0][0]) == 3
    # 101: [CLS]  102: [SEP]  103: [MASK]

    all_graphs = []
    gcn_offsets = []
    for i, sent_token in enumerate(tokens_NoPUNC):
        # 不一定是len(sent_token)-1，因为后面可能补0
        SEPid = sent_token.index(tokenizer.convert_tokens_to_ids('[SEP]'))

        # 去掉所有 '##' 和 '[CLS]'
        sent = ' '.join(re.sub("[#]", "", token) for token in tokenizer.convert_ids_to_tokens(sent_token[1:SEPid]))
        assert len(sent.split(' ')) == SEPid-1

        doc = parser(sent)  # 将句子切分成单词，英文中一般使用空格分隔
        parse_rst = doc.to_json()  # 获得句子中各个单词间的依存关系树

        # 目标代词、A、B [69, 71, 74]
        target_offset_list = [item - 1 for item in offsets_NoPUNC[i]]  # 所有的偏移都去掉一个（[CLS]）

        # 单词
        # # word_id -> node_id，两种id并不一致
        nodes = collections.OrderedDict()  # 带有顺序的字典 key为句子中的id，value为节点的真实索引
        edges = []
        edge_type = []

        #  通过  parse_rst['tokens'][69]可以看到详细信息
        # 解析依存关系
        # {'id': 0, 'start': 0, 'end': 4, 'tag': 'IN', 'pos': 'SCONJ', 'morph': '', 'lemma': 'upon', 'dep': 'prep', 'head': 11}
        # {'id': 1, 'start': 5, 'end': 10, 'tag': 'PRP$', 'pos': 'PRON', 'morph': 'Number=Plur|Person=3|Poss=Yes|PronType=Prs', 'lemma': 'their', 'dep': 'poss', 'head': 2}
        for i_word, word in enumerate(parse_rst['tokens']):
            # 生成的图中，找到代词节点、A、B以及含有它们的边
            if (i_word in target_offset_list) or (word['head'] in target_offset_list):
                if i_word not in nodes:
                    nodes[i_word] = len(nodes)  # 添加依存关系节点
                    edges.append([i_word, i_word])  # 为节点添加自环
                    edge_type.append(0)  # 自环关系的索引为0
                if word['head'] not in nodes:
                    nodes[word['head']] = len(nodes)  # 添加依存关系节点
                    edges.append([word['head'], word['head']])  # 为节点添加自环
                    edge_type.append(0)

                if word['dep'] != 'ROOT':
                    edges.append([word['head'], word['id']])  # 添加依存关系边（head_id 被依赖 -> word_id 依赖）
                    edge_type.append(1)  # 依存关系的索引为1
                    edges.append([word['id'], word['head']])  # 添加反向依存关系边（head_id <- word_id）
                    edge_type.append(2)  # 反向依存关系的索引为2

        # word_id -> node_id，两种id并不一致
        # nodes: OrderedDict([(69, 0), (65, 1), (71, 2), (70, 3), (72, 4), (74, 5), (75, 6)])
        # word_id<-head_id: 69<-65, 71<-70, 71->72, 74<-75
        tran_edges = []
        # 自环边 + 双向边 [[1, 0], [0, 1], [3, 2], [2, 3], [2, 4], [4, 2], [6, 5], [5, 6]]
        for e1, e2 in edges:  # 将句子中的边，换成节点间的边
            tran_edges.append([nodes[e1], nodes[e2]])
        # 将句子中的代词位置，换成节点中的代词索引
        gcn_offset = [nodes[offset] for offset in target_offset_list]
        gcn_offsets.append(gcn_offset)  # 将代词、名称A、名称B对应图中节点的索引保存起来

        # 生成DGL图数据
        G = dgl.DGLGraph()
        G.add_nodes(len(nodes))  # 生成DGL节点
        G.add_edges(list(zip(*tran_edges))[0], list(zip(*tran_edges))[1])
        # 给每个节点添加特征属性
        for i_word, word in nodes.items():
            # 75, 6
            if (i_word in target_offset_list):  # 从PROPN_bert中获取代词、名称A、名称B的特征
                # [0]: bert输出多一维
                G.nodes[[nodes[i_word]]].data['h'] = torch.from_numpy(
                    PROPN_bert[i][0][target_offset_list.index(i_word)]).unsqueeze(0).to(device)
            else:  # bert_forNoPUNC中获取其它词的特征
                # +1: 因为头部的[CLS]
                G.nodes[[nodes[i_word]]].data['h'] = torch.from_numpy(
                    bert_forNoPUNC[i][0][i_word + 1]).unsqueeze(0).to(device)

        edge_norm = []  # 归一化算子（计算均值时的分母）
        # 按图的边索引遍历 e1->e2
        for e1, e2 in tran_edges:
            if e1 == e2:
                edge_norm.append(1)  # 如果是自环边，则归一化算子为1
            else:  # 如果是非自环边，则归一化算子为1除以去掉自环的度
                edge_norm.append(1 / (G.in_degrees(e2) - 1))  # 去掉自环的度

        # 将类型转为张量
        edge_type = torch.from_numpy(np.array(edge_type)).type(torch.long)  # uint8 会导致错误
        edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(device)

        G.edata.update({'rel_type': edge_type, })  # 更新边特征
        G.edata.update({'norm': edge_norm})
        all_graphs.append(G)  # 保存子图

    return all_graphs, gcn_offsets


def getLabelData(df):  # 生成标签
    tmp = df[["A-coref", "B-coref"]].copy()
    tmp["Neither"] = ~(df["A-coref"] | df["B-coref"])  # 添加一个列（A和B都不指代的情况）
    y = tmp.values.astype("bool").argmax(1)  # 变成one-hot索引
    return y


########################################################################


# 构建数据集
class GPRDataset(Dataset):
    def __init__(self, y, graphs, bert_offsets, gcn_offsets, bert_embeddings):
        self.y = y
        self.graphs = graphs
        self.bert_offsets = bert_offsets  # 已经+1了
        self.gcn_offsets = gcn_offsets  # 图中代词及名称AB的节点索引
        self.bert_embeddings = bert_embeddings  # 有[CLS]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (self.graphs[idx], self.bert_offsets[idx], self.gcn_offsets[idx],
                self.bert_embeddings[idx], self.y[idx])


# collate: 整理
def collate(samples):  # 对批次数据重新加工
    #    print(len(samples))#数组。个数是4（batch_size），

    # 行列转换变成list
    graphs, bert_offsets, gcn_offsets, bert_embeddings, labels = map(list, zip(*samples))

    # 合并图，要搭配dgl.unbatch()
    batched_graph = dgl.batch(graphs)  # 对图数据进行按批次重组 !!!批次介绍！！
    # 对其它数据进行张量转化
    offsets_bert = torch.stack([torch.LongTensor(x) for x in bert_offsets], dim=0)
    offsets_gcn = torch.stack([torch.LongTensor(x) for x in gcn_offsets], dim=0)
    bert_embeddings = torch.from_numpy(np.asarray(bert_embeddings))
    one_hot_labels = torch.from_numpy(np.asarray(labels)).type(torch.long)  # .squeeze()#必须要用long

    return batched_graph, offsets_bert, offsets_gcn, bert_embeddings, one_hot_labels


# 将训练数据集转化为图数据
all_graphs, gcn_offsets = getGraphsData(tokens_NoPUNC, offsets_NoPUNC, PROPN_bert, bert_forNoPUNC)
train_y = getLabelData(df_train_val)  # 获取训练数据集的标签


# 将测试数据集转化为图数据
test_all_graphs, test_gcn_offsets = getGraphsData(test_tokens_NoPUNC, test_offsets_NoPUNC,
                                                  test_PROPN_bert, test_bert_forNoPUNC)
test_y = getLabelData(df_test)  # 获取测试数据集的标签
# 生成测试数据集
test_dataset = GPRDataset(test_y, test_all_graphs, test_offsets_NoPUNC,
                          test_gcn_offsets, test_PROPN_bert)
# 生成测试数据集的加载器
test_dataloarder = DataLoader(test_dataset, collate_fn=collate, batch_size=4)


#########################
# 构建模型

class RGCNModel(nn.Module):  # 多层R-GCN模型
    def __init__(self, h_dim, num_rels, out_dim=256, num_hidden_layers=1):
        super(RGCNModel, self).__init__()

        self.layers = nn.ModuleList()  # 定义网络层列表

        for _ in range(num_hidden_layers):
            rgcn_layer = RelGraphConv(h_dim, out_dim, num_rels, activation=F.relu)
            self.layers.append(rgcn_layer)

    def forward(self, g):
        # 逐层处理
        for layer in self.layers:
            # RelGraphConv.forward(self, g, feat, etypes, norm=None, *, presorted=False)
            g.ndata['h'] = layer(g, g.ndata['h'].to(device), etypes=g.edata['rel_type'].to(device),
                                 norm=g.edata['norm'].to(device))

        rst_hidden = []
        # 搭配dgl.batch()
        for sub_g in dgl.unbatch(g):  # 按批次解包
            rst_hidden.append(sub_g.ndata['h'])
        return rst_hidden


# Design the Main Model (R-GCN + FFNN)
class BERT_Head(nn.Module):
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_hidden_size * 3),
            nn.Dropout(0.5),
            nn.Linear(bert_hidden_size * 3, 512 * 3),
            nn.ReLU(),
        )

        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, bert_embeddings):
        # print('BERT_Head bert_embeddings: ', bert_embeddings, bert_embeddings.view(bert_embeddings.shape[0],-1).shape)
        outputs = self.fc(bert_embeddings.view(bert_embeddings.shape[0], -1))
        return outputs


class Head(nn.Module):
    """The MLP submodule"""

    def __init__(self, gcn_out_size: int, bert_out_size: int):
        super().__init__()
        self.bert_out_size = bert_out_size
        self.gcn_out_size = gcn_out_size

        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_out_size * 3 + gcn_out_size * 3),
            nn.Dropout(0.5),
            nn.Linear(bert_out_size * 3 + gcn_out_size * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )
        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, gcn_outputs, offsets_gcn, bert_embeddings):

        gcn_extracted_outputs = [gcn_outputs[i].unsqueeze(0).gather(1, offsets_gcn[i].unsqueeze(0).unsqueeze(2)
                                                                    .expand(-1, -1,
                                                                            gcn_outputs[i].unsqueeze(0).size(2))).view(
            gcn_outputs[i].unsqueeze(0).size(0), -1) for i in range(len(gcn_outputs))]

        gcn_extracted_outputs = torch.stack(gcn_extracted_outputs, dim=0).squeeze()

        embeddings = torch.cat((gcn_extracted_outputs, bert_embeddings), 1)

        return self.fc(embeddings)


class GPRModel(nn.Module):
    """The main model."""

    def __init__(self):
        super().__init__()
        self.RGCN = RGCNModel(h_dim=768, out_dim=256, num_rels=3)
        self.BERThead = BERT_Head(768)  # bert output size
        self.head = Head(256, 512)  # gcn output   berthead output

    def forward(self, offsets_bert, offsets_gcn, bert_embeddings, g):
        gcn_outputs = self.RGCN(g)
        bert_head_outputs = self.BERThead(bert_embeddings)
        head_outputs = self.head(gcn_outputs, offsets_gcn, bert_head_outputs)
        return head_outputs


def adjust_learning_rate(optimizers, epoch, lr_value):
    '''
    在训练过程的不同阶段动态地调整优化器的学习率.
    通常用于训练深度神经网络，目的是在训练的不同阶段采用不同的学习率，以提高模型的收敛速度和性能。
    Args:
        optimizers:
        epoch:
        lr_value:

    Returns:

    '''
    # warm up
    if epoch < 10:
        lr_tmp = 0.00001
    else:
        lr_tmp = lr_value * pow((1 - 1.0 * epoch / 100), 0.9)
    if epoch > 36:
        lr_tmp = 0.000015 * pow((1 - 1.0 * epoch / 100), 0.9)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_tmp
    return lr_tmp


def trainmodel(train_dataloarder, val_dataloarder, model, loss_func, optimizer, lr_value):
    reg_lambda = 0.035
    total_epoch = 100
    best_val_loss = 11
    # Cross-Entropy
    ce_losses = []
    epoch_losses = []
    val_losses = []
    val_acclist = []
    for epoch in range(total_epoch):

        if epoch % 5 == 0:
            print('|', ">" * epoch, " " * (80 - epoch), '|')

        lr = adjust_learning_rate([optimizer], epoch, lr_value)
        print("Learning rate = %4f\n" % lr)
        model.train()
        epoch_loss = 0
        reg_loss = 0
        ce_loss = 0
        for iter, (batched_graph, offsets_bert, offsets_gcn, bert_embeddings, labels) in enumerate(train_dataloarder):

            bert_embeddings = bert_embeddings.to(device)
            labels = labels.to(device)
            offsets_gcn = offsets_gcn.to(device)
            # batched_graph g.batch_size 4,g.batch_num_nodes [6, 6, 8, 6],g.batch_num_edges[12, 14, 20, 16]
            prediction = model(offsets_bert, offsets_gcn, bert_embeddings, batched_graph)
            l2_reg = None
            for w in model.RGCN.parameters():
                if not l2_reg:
                    l2_reg = w.norm(2)
                else:
                    l2_reg = l2_reg + w.norm(2)
            for w in model.head.parameters():
                if not l2_reg:
                    l2_reg = w.norm(2)
                else:
                    l2_reg = l2_reg + w.norm(2)
            for w in model.BERThead.parameters():
                if not l2_reg:
                    l2_reg = w.norm(2)
                else:
                    l2_reg = l2_reg + w.norm(2)
            loss = loss_func(prediction, labels) + l2_reg * reg_lambda
            # loss = loss_func(prediction, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###########################
            epoch_loss += loss.detach().item()
            reg_loss += (l2_reg * reg_lambda).detach().item()
            ce_loss += (loss_func(prediction, labels)).detach().item()
        epoch_loss /= (iter + 1)
        ce_loss /= (iter + 1)
        reg_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}, ce_loss {:.4f}, reg_loss {:.4f}'.format(epoch, epoch_loss, ce_loss, reg_loss))
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
        ce_losses.append(ce_loss)
        ##################################

        val_loss = 0
        model.eval()
        val_accs = []
        for iter, (batched_graph, offsets_bert, offsets_gcn, bert_embeddings, labels) in enumerate(val_dataloarder):
            offsets_gcn = offsets_gcn.to(device)
            bert_embeddings = bert_embeddings.to(device)
            labelsgpu = labels.to(device)
            with torch.no_grad():
                prediction = model(offsets_bert, offsets_gcn, bert_embeddings, batched_graph)
            loss = loss_func(prediction, labelsgpu)
            val_loss += loss.detach().item()

            val_acc = metrics.accuracy_score(labels, torch.argmax(prediction, -1).cpu().numpy())
            val_accs.append(val_acc)

        val_loss = val_loss / (iter + 1)
        val_losses.append(val_loss)
        val_acclist.append(np.mean(val_accs))

        if epoch % 20 == 0:
            print('Epoch {}, val_loss {:.4f}, val_acc {:.4f}'.format(epoch,
                                                                     val_loss, np.mean(val_accs)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if epoch > 20:
                torch.save(model.state_dict(), base_dir + 'best_model.pth')
            if epoch > 36: print('Best val loss found: ', best_val_loss)

        ################
        print('Epoch {}, val_loss {:.4f}, val_acc {:.4f}'.format(epoch,
                                                                 val_loss, np.mean(val_accs)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    #########################
    print('This fold, the best val loss is: ', best_val_loss)
    return ce_losses, val_losses, val_acclist


# 5 fold

kfold = StratifiedKFold(n_splits=5)


def getdataloader(index, isshuffle=False):
    dataset = GPRDataset(train_y[index],
                         list(itemgetter(*index)(all_graphs)),
                         list(itemgetter(*index)(offsets_NoPUNC)),
                         list(itemgetter(*index)(gcn_offsets)),
                         list(itemgetter(*index)(PROPN_bert)))
    dataloarder = DataLoader(dataset, collate_fn=collate,
                             batch_size=4, shuffle=isshuffle)

    return dataloarder


test_predict_lst = []  # the test output for every fold
for train_index, test_index in kfold.split(df_train_val, train_y):  # 循环5次
    print("=" * 20)
    print(f"Fold {len(test_predict_lst) + 1}")
    print("=" * 20)

    val_dataloarder = getdataloader(test_index)
    train_dataloarder = getdataloader(train_index, True)
    print('Dataloader Success---------------------')

    model = GPRModel().to(device)
    loss_func = nn.CrossEntropyLoss()
    lr_value = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr_value)
    ce_losses, val_losses, val_accs = trainmodel(train_dataloarder,
                                                 val_dataloarder,
                                                 model, loss_func, optimizer, lr_value)

    plt.figure()
    plt.plot(ce_losses, label='CE_loss')
    plt.plot(val_losses, label='Val_loss')
    plt.plot(val_accs, label='Val_acc')
    plt.legend()  # 添加图例
    plt.show()

    # 测试
    test_loss = 0.
    test_predict = None
    model.load_state_dict(torch.load(base_dir + 'best_model.pth'))
    model.to(device)
    model.eval()

    for iter, (batched_graph, offsets_bert, offsets_gcn, bert_embeddings,
               labels) in enumerate(test_dataloarder):

        offsets_gcn = offsets_gcn.to(device)
        bert_embeddings = bert_embeddings.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            prediction = model(offsets_bert, offsets_gcn, bert_embeddings, batched_graph)

        if test_predict is None:
            test_predict = prediction
        else:
            test_predict = torch.cat((test_predict, prediction), 0)
        loss = loss_func(prediction, labels)
        test_loss += loss

    acc = metrics.accuracy_score(test_y, torch.argmax(test_predict, -1).cpu().numpy())
    test_loss /= (iter + 1)
    print('This fold, the test loss is: ', test_loss, " acc is ", acc)
    test_predict_lst.append(test_predict)

# Test Part
test_predict_arr = [torch.softmax(pre.cpu(), -1).clamp(1e-4, 1 - 1e-4).numpy() for pre in test_predict_lst]
final_test_preds = np.mean(test_predict_arr, axis=0)


def extract_target(df):
    df["Neither"] = 0
    df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1
    df["target"] = 0
    df.loc[df['B-coref'] == 1, "target"] = 1
    # 满足条件 df["Neither"] == 1 的行的 "target" 列的值设置为2，"target" 列不会自动创建
    df.loc[df["Neither"] == 1, "target"] = 2
    return df


test_df = extract_target(df_test)
log_loss(test_df.target, final_test_preds)

# 'result' 预测结果
result = np.argmax(final_test_preds, -1).reshape(len(final_test_preds), 1)

# 保存结果
df_sub = pd.DataFrame(np.concatenate([final_test_preds, result], -1), columns=["A", "B", "NEITHER", 'result'])
df_sub["ID"] = test_df.ID
df_sub["target"] = test_df["target"]
df_sub = df_sub[['ID', "A", "B", "NEITHER", "result", "target"]]
print('df_sub')
print(df_sub.head(50))
df_sub.to_csv(base_dir + "submission_415_copy3.csv", index=False)

acc = metrics.accuracy_score(test_df["target"].values, np.argmax(final_test_preds, -1))
print('acc', acc)
