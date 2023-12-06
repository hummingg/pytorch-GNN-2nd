# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 07:10:37 2020
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""

# 提取代词特征

import pandas as pd
import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig

# 指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 读取数据
# 读取数据 变量命名有问题？
df_test = pd.read_csv("gap-coreference/gap-development.tsv", delimiter="\t")
df_train_val = pd.concat([
    pd.read_csv("gap-coreference/gap-test.tsv", delimiter="\t"),
    pd.read_csv("gap-coreference/gap-validation.tsv", delimiter="\t")
], axis=0)


def getmodel():
    # 加载词表文件tokenizer
    tokenizer = BertTokenizer.from_pretrained('huggingface/bert-base-uncased')

    # 添加特殊词
    special_tokens_dict = {'additional_special_tokens': ["[THISISA]", "[THISISB]", "[THISISP]"]}
    tokenizer.add_special_tokens(special_tokens_dict)  # 添加特殊词
    print(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)

    model = BertModel.from_pretrained('huggingface/bert-base-uncased')  # 加载模型
    return tokenizer, model


############################


def insert_tag(row, hasbrack=True):  # 按照插入的位置，从大到小排序[(383, ' THISISP '), (366, ' THISISB '), (352, ' THISISA ')]
    orgtag = [" [THISISA] ", " [THISISB] ", " [THISISP] "]
    if hasbrack == False:
        orgtag = [" THISISA ", " THISISB ", " THISISP "]

    to_be_inserted = sorted([
        (row["A-offset"], orgtag[0]),
        (row["B-offset"], orgtag[1]),
        (row["Pronoun-offset"], orgtag[2])], key=lambda x: x[0], reverse=True)

    text = row["Text"]  # len 443
    for offset, tag in to_be_inserted:  # 先插最后的，不会影响前面
        text = text[:offset] + tag + text[offset:]  # （插到每个代词的前面）
    return text  # len 470 (443+3*9)


def tokenize(sequence_ind, tokenizer, sequence_mask=None):  # 将标签分离，并返回标签偏移位置
    entries = {}
    final_tokens = []
    final_mask = []

    for i, one in enumerate(sequence_ind):
        if one in tokenizer.additional_special_tokens_ids:
            tokenstr = tokenizer.convert_ids_to_tokens(one)
            entries[tokenstr] = len(final_tokens)
            continue
        final_tokens.append(one)
        if sequence_mask is not None:
            final_mask.append(sequence_mask[i])
    return final_tokens, (entries["[THISISA]"], entries["[THISISB]"], entries["[THISISP]"]), final_mask


def savepkl(df, name):
    bert_prediction = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 循环内部
        text = insert_tag(row)  # 插入标签
        # 97
        sequence_ind = tokenizer.encode(text)  # 向量化
        # 94, 3. 42+1->42:30522 46+1->45:30523 64+1->62:30524 ?按前后顺序一次删掉特殊标记
        tokens, offsets, _ = tokenize(sequence_ind, tokenizer)  # 获取标签偏移
        token_tensor = torch.LongTensor([tokens]).to(device)
        # 错误旧版写法
        # bert_outputs,bert_last_outputs =  model(token_tensor)  #[1, 94, 768] , [1, 768]
        # res = model(token_tensor)
        # bert_outputs, bert_last_outputs = res['last_hidden_state'], res['pooler_output']
        bert_outputs, bert_last_outputs = model(token_tensor)[:2]  # [1, 94, 768] , [1, 768]
        # (1, 3, 768)
        extracted_outputs = bert_outputs[:, offsets, :]  # 根据偏移位置抽取特征向量
        bert_prediction.append(extracted_outputs.cpu().numpy())
    # 所有样本的 1代词+2名称 的BERT向量, 保存起来
    pickle.dump(bert_prediction, open(name, "wb"))


if __name__ == '__main__':
    tokenizer, model = getmodel()
    model.to(device)
    torch.set_grad_enabled(False)

    # 找不到文件？在/tmp/pycharm_project_pytorch-GNN
    savepkl(df_test, 'output/test_bert_outputs_forPROPN.pkl')
    savepkl(df_train_val, 'output/bert_outputs_forPROPN.pkl')
