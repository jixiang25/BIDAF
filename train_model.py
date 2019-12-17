import argparse
import json
import pickle
import torch
import torch.nn as nn
import os

from dataset import SquadDataset
from torch.utils.data import DataLoader
from model import BIDAF
from loss import MCLoss
from utils import predict, exact_match, f1
from ema import EMA
from tqdm import tqdm


def train_epoch(model, criterion, optimizer, ema, train_loader, max_grad_norm, decay_rate):
    model.train()
    device = model.device

    train_loss, train_em, train_f1 = 0.0, 0.0, 0.0

    tqdm_train_loader = tqdm(train_loader)

    for idx, batch in enumerate(tqdm_train_loader):
        context = batch['context'].to(device)
        context_length = batch['context_length'].to(device)
        question = batch['question'].to(device)
        question_length = batch['question_length'].to(device)
        answer = batch['answer']

        optimizer.zero_grad()

        prob_start, prob_end = model(context, context_length, question, question_length)
        # loss = criterion(prob_start, answer[0]) + criterion(prob_end, answer[1])
        loss = criterion(prob_start, prob_end, answer)
        loss.backward()

        # nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        train_loss += loss.item()

        predict_answer = predict(prob_start, prob_end, context_length)
        # print(predict_answer, answer)
        train_em += exact_match(predict_answer, answer)
        train_f1 += f1(predict_answer, answer, context)

        description = 'Avg. batch train loss:{:.6f}'.format(train_loss / (idx + 1))
        tqdm_train_loader.set_description(description)

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = ema(name, param.data)

    train_loss = train_loss / len(train_loader)
    train_em = train_em / len(train_loader.dataset)
    train_f1 = train_f1 / len(train_loader.dataset)

    return train_loss, train_em, train_f1


def valid_epoch(model, criterion, dev_loader):
    model.eval()
    device = model.device

    valid_loss, valid_em, valid_f1 = 0, 0, 0
    tqdm_dev_loader = tqdm(dev_loader)

    for idx, batch in enumerate(tqdm_dev_loader):
        context = batch['context'].to(device)
        context_length = batch['context_length'].to(device)
        question = batch['question'].to(device)
        question_length = batch['question_length'].to(device)
        answer = batch['answer']

        prob_start, prob_end = model(context, context_length, question, question_length)
        # loss = criterion(prob_start, answer[0]) + criterion(prob_end, answer[1])
        loss = criterion(prob_start, prob_end, answer)
        valid_loss += loss.item()

        predict_answer = predict(prob_start, prob_end, context_length)
        valid_em += exact_match(predict_answer, answer)
        valid_f1 += f1(predict_answer, answer, context)

        description = 'Avg. batch dev loss: {:.6f}'.format(valid_loss / (idx + 1))
        tqdm_dev_loader.set_description(description)
    valid_loss = valid_loss / len(dev_loader)
    valid_em = valid_em / len(dev_loader.dataset)
    valid_f1 = valid_f1 / len(dev_loader.dataset)
    return valid_loss, valid_em, valid_f1


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
    dev_loader = DataLoader(squad_dev_data, shuffle=False, batch_size=batch_size)

    print('=' * 20, 'Loading embedding matrix', '=' * 20)
    with open(embedding_dir, 'rb') as embedding_file:
        embedding = torch.tensor(pickle.load(embedding_file), dtype=torch.float).to(device)

    model = BIDAF(vocab_size=embedding.size()[0],
                  embedding_dim=embedding.size()[1],
                  embedding=embedding,
                  dropout=dropout,
                  device=device,
                  padding_idx=0).to(device)

    criterion = MCLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    ema = EMA(decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    print('\n', '=' * 20, 'Train BIDAF model on device: {}'.format(device), '=' * 20, '\n')

    best_f1 = 0.0
    patience_count = 0
    for epoch in range(1, epochs + 1):

        train_loss, train_em, train_f1 = train_epoch(model=model,
                                                     criterion=criterion,
                                                     optimizer=optimizer,
                                                     ema=ema,
                                                     train_loader=train_loader,
                                                     max_grad_norm=max_grad_norm,
                                                     decay_rate=decay_rate)
        print('* Train epoch {}'.format(epoch))
        print('-> loss={}, em={}, f1={}'.format(train_loss, train_em, train_f1))

        dev_loss, dev_em, dev_f1 = valid_epoch(model=model,
                                               criterion=criterion,
                                               dev_loader=dev_loader)
        print('* Valid epoch {}'.format(epoch))
        print('-> loss={}, em={}, f1={}'.format(dev_loss, dev_em, dev_f1))

        if dev_f1 > best_f1:
            torch.save({'model': model.state_dict()},
                       os.path.join(checkpoint_dir, 'bidaf_{}.pth.tar'.format(epoch)))
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_max:
                print('-> Early stopping at epoch {}...'.format(epoch))
                break


def main():
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


if __name__ == '__main__':
    main()
