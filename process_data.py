import argparse
import json
import numpy as np
import pickle


def process_squad_data(train_dir, dev_dir):
    # 得到train_set, 以及word_dict
    train = {
        'context': list(),
        'question': list(),
        'answer': list()
    }
    word_dict = {
        '__PAD__': 0,
        '__OOV__': 1,
        '__BOS__': 2,
        '__EOS__': 3
    }
    word_dict_size = 4
    with open(train_dir, 'r') as train_file:
        train_data = json.loads(train_file.read())
    for dt_pkg in train_data:
        train['context'].append(dt_pkg['context'])
        train['question'].append(dt_pkg['question'])
        train['answer'].append(dt_pkg['answer'])
        for word in dt_pkg['context']:
            if word_dict.get(word) is None:
                word_dict[word] = word_dict_size
                word_dict_size += 1
        for word in dt_pkg['question']:
            if word_dict.get(word) is None:
                word_dict[word] = word_dict_size
                word_dict_size += 1

    # 得到dev_set
    dev = {
        'context': list(),
        'question': list(),
        'answer': list()
    }
    with open(dev_dir, 'r') as dev_file:
        dev_data = json.loads(dev_file.read())
    for dt_pkg in dev_data:
        dev['context'].append(dt_pkg['context'])
        dev['question'].append(dt_pkg['question'])
        dev['answer'].append(dt_pkg['answer'])

    return train, dev, word_dict


def transfer_index_data(data, word_dict):
    index_data = {
        'context': [],
        'question': [],
        'answer': []
    }
    for context_list, question_list in zip(data['context'], data['question']):
        index_context_list = [word_dict['__BOS__']]
        for word in context_list:
            index_context_list.append(word_dict[word] if word in word_dict else word_dict['__OOV__'])
        index_context_list.append(word_dict['__EOS__'])
        index_question_list = [word_dict['__BOS__']]
        for word in question_list:
            index_question_list.append(word_dict[word] if word in word_dict else word_dict['__OOV__'])
        index_question_list.append(word_dict['__EOS__'])
        index_data['context'].append(index_context_list)
        index_data['question'].append(index_question_list)
    index_data['answer'] = [[answer[0] + 1, answer[1] + 1] for answer in data['answer']]
    return index_data


def build_embedding_matrix(word_dict, pretrained_embedding_dir, word_embedding_dim):
    glove_dict = dict()
    with open(pretrained_embedding_dir, 'r', encoding='utf-8') as f:
        while True:
            word_info = f.readline()
            if len(word_info) <= 0:
                break
            word_info = word_info.rstrip().split()
            word, word_numbers = word_info[0], word_info[1:]
            word_numbers = [float(word_num) for word_num in word_numbers]
            glove_dict[word] = word_numbers
    print('num words in glove embedding: ', len(glove_dict))
    word_dict_len = len(word_dict)
    embedding_matrix = np.zeros((word_dict_len, word_embedding_dim))
    seen_word = 0
    for word, idx in word_dict.items():
        if word in glove_dict:
            embedding_matrix[idx] = np.array(glove_dict[word], dtype=float)
            seen_word += 1
        else:
            if word == '__PAD__':
                continue
            embedding_matrix[idx] = np.random.normal(size=word_embedding_dim)
    print('seen_word numbers: ', seen_word)
    return embedding_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/process_data.config')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = json.loads(config_file.read())
    train, dev, word_dict = process_squad_data(train_dir=config['train_dir'], dev_dir=config['dev_dir'])
    with open(config['output_word_dict_dir'], 'w') as f:
        f.write(json.dumps(word_dict, indent=2))
    index_train= transfer_index_data(data=train, word_dict=word_dict)
    index_dev = transfer_index_data(data=dev, word_dict=word_dict)
    embedding_matrix = build_embedding_matrix(word_dict=word_dict,
                                              pretrained_embedding_dir=config['word_embedding_dir'],
                                              word_embedding_dim=config['word_embedding_dim'])
    with open(config['output_train_dir'], 'w') as f:
        f.write(json.dumps(index_train, indent=2))
    with open(config['output_dev_dir'], 'w') as f:
        f.write(json.dumps(index_dev, indent=2))
    with open(config['output_embedding_matrix'], 'wb') as embedding_pkl:
        pickle.dump(embedding_matrix, embedding_pkl)


if __name__ == '__main__':
    main()
