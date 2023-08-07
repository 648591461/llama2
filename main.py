import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 定义字典
words = '<PAD>,<BOS>,<EOS>,1,2,3,4,5,6,7,8,9,0,+,='
vocab = {word: i for i, word in enumerate(words.split(','))}  # 语料
vocab_r = [k for k, v in vocab.items()]  # 反向查询


# 两数相加数据集
def get_data(min_length=10, max_length=20):
    # 定义词的集合
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 每个词被选中的概率
    p = np.array([7, 5, 5, 7, 6, 5, 7, 6, 5, 7])
    p = p / p.sum()

    # 随机采样n1个词作为s1, 即加数 A
    n1 = random.randint(min_length, max_length)
    s1 = np.random.choice(words, size=n1, replace=True, p=p)
    s1 = s1.tolist()

    # 随机采样n2个词作为s2, 即加数 B
    n2 = random.randint(min_length, max_length)
    s2 = np.random.choice(words, size=n2, replace=True, p=p)
    s2 = s2.tolist()

    # x等于s1和s2字符上的相加
    x = s1 + ['+'] + s2 + ['=']

    y = int(''.join(s1)) + int(''.join(s2))
    y = list(str(y))

    # 加上起始符与终止符
    x = ['<BOS>'] + x
    y = y + ['<EOS>']
    return x, y


# 定义数据集
class TwoSumDataset(torch.utils.data.Dataset):
    def __init__(self, size=100000, min_length=10, max_length=20):
        super(Dataset, self).__init__()
        self.size = size
        self.min_length = min_length
        self.max_length = max_length

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        x, y = self.get(i)

        # 编码成token
        context_ids = [vocab[i] for i in x]
        target_ids = [vocab[i] for i in y]

        input_ids = context_ids + target_ids

        # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss
        labels = [-100] * len(context_ids) + target_ids
        masks = [0 if t == vocab['<PAD>'] else 1 for t in input_ids]
        example = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': masks}

        return example

    def get(self, i):
        return get_data(self.min_length, self.max_length)


    def show_example(self, example):
        input_ids, labels = example['input_ids'],example['labels']
        x = ''.join([vocab_r[a] for a, b in zip(input_ids, labels) if b == -100])
        y = ''.join([vocab_r[a] for a, b in zip(input_ids, labels) if b != -100])
        print(x+y)


ds_train = TwoSumDataset(size=100000, min_length=10, max_length=20)
ds_val = TwoSumDataset(size=10000, min_length=10, max_length=20)


def data_collator(examples: list):
    len_ids = [len(example['input_ids']) for example in examples]
    longest = max(len_ids)  # 以训练集中最长的input_ids为准，对其他的input_ids进行padding

    input_ids = []
    labels_list = []
    masks_list = []

    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example['input_ids']
        labs = example['labels']
        masks = example['attention_mask']

        ids = [vocab['<PAD>']] * (longest - length) + ids  # 为什么倒着填充？
        labs = [-100] * (longest - length) + labs
        masks = [0] * (longest - length) + masks

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))
        masks_list.append(torch.LongTensor(masks))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(masks_list)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask":attention_mask
    }

# 数据加载器
# 数据加载器
dl_train = DataLoader(dataset=ds_train,
         batch_size=200,
         drop_last=True,
         shuffle=True,
         collate_fn = data_collator
        )

dl_val = DataLoader(dataset=ds_val,
         batch_size=200,
         drop_last=True,
         shuffle=False,
         collate_fn = data_collator
        )
