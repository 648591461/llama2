# 定义字典 与 预料
words = '<PAD>,<BOS>,<EOS>,1,2,3,4,5,6,7,8,9,0,+,='
vocab = {word: i for i, word in enumerate(words.split(','))}  # 语料
vocab_r = [k for k, v in vocab.items()]  # 反向查询