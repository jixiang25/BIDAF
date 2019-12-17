import torch
import pickle
from model import BIDAF


def test1():
    with open('./data/glove.6B.200d.txt', 'r', encoding='utf-8') as f:
        cnt = 0
        while True:
            try:
                word_info = f.readline()
            except:
                print(cnt)
                raise ValueError('Stop!')
            if len(word_info) <= 0:
               break
            cnt += 1
            word = word_info.rstrip().split()
            w, word = word[0], word[1:]
            word = [float(number) for number in word]
            # print(w, word)


def test2():
    from layers import Similarity, Context2Query, Query2Context
    sim = Similarity(9)
    c2q = Context2Query()
    q2c = Query2Context()
    batch_size = 2
    c_len = 5
    q_len = 4
    dim = 3
    question = torch.randn((2, 4, 3), dtype=torch.float)
    context = torch.randn((2, 5, 3), dtype=torch.float)
    context_length,question_length = [4, 5], [3, 2]
    context[0, 4:5, :] = 0.0
    question[0, 3:4, :] = 0.0
    question[1, 2:4, :] = 0.0
    # print(context)
    # print(question)
    simi, mask = sim(context, context_length, question, question_length)
    # print(simi)
    context_query = c2q(simi, question, mask)
    query_context = q2c(simi, context, mask, context_query)
    print(context_query.size())
    print(query_context.size())


def test3():
    from layers import OutputLayer
    l = OutputLayer(10)
    x = torch.randn(3, 5, 10)
    context_length = torch.tensor([4, 5, 3])
    output = l(x, context_length)
    print(output)
    print(output.size())


def test4():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/train_model.config')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = json.loads(config_file.read())
    train(train_dir=config['train_dir'],
          dev_dir=config['dev_dir'],
          embedding_dir=config['embedding_dir'],
          checkpoint_dir=config['checkpoint_dir'],
          context_max_len=config['context_max_len'],
          question_max_len=config['question_max_len'],
          batch_size=config['batch_size'],
          epochs=config['epochs'],
          learning_rate=config['learning_rate'],
          decay_rate=config['decay_rate'],
          max_grad_norm=config['max_grad_norm'],
          patience_max=config['patience_max'],
          dropout=config['dropout'])


def train(train_dir,
          dev_dir,
          embedding_dir,
          checkpoint_dir,
          context_max_len=100,
          question_max_len=30,
          batch_size=16,
          epochs=20,
          learning_rate=0.0004,
          decay_rate=0.999,
          max_grad_norm=10.0,
          patience_max=5,
          dropout=0.2):
    from dataset import SquadDataset
    from torch.utils.data import DataLoader
    import json
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    print('=' * 20, 'Preparing for train data', '=' * 20)
    with open(train_dir, 'r') as train_file:
        train_data = json.loads(train_file.read())
    squad_train_data = SquadDataset(data=train_data,
                                    padding_idx=0,
                                    context_max_len=context_max_len,
                                    question_max_len=question_max_len)
    train_loader = DataLoader(squad_train_data, shuffle=False, batch_size=batch_size)

    print('=' * 20, 'Preparing for train data', '=' * 20)
    with open(dev_dir, 'r') as train_file:
        dev_data = json.loads(train_file.read())
    squad_dev_data = SquadDataset(data=dev_data,
                                  padding_idx=0,
                                  context_max_len=context_max_len,
                                  question_max_len=question_max_len)
    dev_loader = DataLoader(squad_train_data, shuffle=False, batch_size=batch_size)
    with open('./data/word_dict.json', 'r') as f:
        word_dict = json.load(f)
    idx_dict = dict()
    for ks, vs in word_dict.items():
        idx_dict[vs] = ks
    # print(word_dict)
    cnt = 0
    for batch in dev_loader:
        cnt += 1
        if cnt < len(dev_loader):
            continue
        context, context_length = batch['context'], batch['context_length']
        question, question_length = batch['question'], batch['question_length']
        answer = batch['answer']
        for batch_idx in range(context.shape[0]):
            context_seq, question_seq, ans = '', '', ''
            for word_idx in range(context_max_len):
                index = context[batch_idx][word_idx].item()
                context_seq += idx_dict[index] + ' '
            for word_idx in range(question_max_len):
                index = question[batch_idx][word_idx].item()
                question_seq += idx_dict[index] + ' '
            answer_start, answer_end = answer[0][batch_idx], answer[1][batch_idx]
            for word_idx in range(answer_start, answer_end + 1):
                index = context[batch_idx][word_idx].item()
                ans += idx_dict[index] + ' '
            print(context_seq)
            print(question_seq)
            print(ans)
            print('\n')
        break


def test5():
    from utils import sort_seq_by_len
    x = torch.randn(5, 5, 2)
    length = torch.tensor([200, 500, 300, 400, 100])
    sorted_batch, sorted_seq_len, reverse_mapping = sort_seq_by_len(x, length)
    print(length)
    print(x)
    print(sorted_batch)
    print(sorted_seq_len)
    print(reverse_mapping)


def test6():
    def __te(a):
        x = a.view(2, 3)
        x[1][1] = 100
    x = torch.arange(0, 6)
    __te(x)
    print(x)


def test7():
    from utils import sort_seq_by_len
    layer = torch.nn.LSTM(
        input_size=2,
        hidden_size=2,
        bidirectional=True,
        batch_first=True
    )
    x = torch.tensor([[[-0.0589,  0.9963],
         [ 0.1615, -0.6034],
         [ 0.9466, -2.0558],
         [ 0.2355, -0.1744],
         [ 0.0911, -0.4407]],

        [[-0.3196,  0.3055],
         [-1.5528,  0.5988],
         [-0.1501,  0.5257],
         [ 1.2415,  0.0069],
         [ 0.8009,  1.8843]],

        [[ 0.2402, -1.4367],
         [ 0.7753, -0.8946],
         [ 0.5594,  0.5835],
         [ 1.5598, -0.5371],
         [ 0.3914,  1.2545]]])
    length = torch.tensor([4, 3, 4])
    sorted_x, sorted_length, restore_idx = sort_seq_by_len(x, length)
    packed_x = torch.nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_length, batch_first=True)
    y, _ = layer(packed_x, None)
    pad_y, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_x, batch_first=True)
    pad_y = pad_y.index_select(0, restore_idx)
    print(pad_y)
    print(pad_y[0].size(), pad_y[1].size(), pad_y[2].size())


def test8():
    from utils import sort_seq_by_len
    layer = torch.nn.LSTM(
        input_size=2,
        hidden_size=2,
        bidirectional=True,
        batch_first=True
    )
    x = torch.tensor([[[-0.0589,  0.9963],
         [ 0.1615, -0.6034],
         [ 0.9466, -2.0558],
         [ 0.2355, -0.1744],
         [ 0.0911, -0.4407]],

        [[-0.3196,  0.3055],
         [-1.5528,  0.5988],
         [-0.1501,  0.5257],
         [ 1.2415,  0.0069],
         [ 0.8009,  1.8843]],

        [[ 0.2402, -1.4367],
         [ 0.7753, -0.8946],
         [ 0.5594,  0.5835],
         [ 1.5598, -0.5371],
         [ 0.3914,  1.2545]]])
    """
    length = torch.tensor([4, 3, 5])
    sorted_x, sorted_length, restore_idx = sort_seq_by_len(x, length)
    packed_x = torch.nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_length, batch_first=True)
    y, _ = layer(packed_x, None)
    pad_y, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_x, batch_first=True)
    pad_y = pad_y.index_select(0, restore_idx)
    """
    pad_y, _ = layer(x, None)
    print(pad_y)
    # print(pad_y[0].size(), pad_y[1].size(), pad_y[2].size())


test4()
