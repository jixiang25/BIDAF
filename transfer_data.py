import argparse
import json


def preprocess_squad_data(train_dir, dev_dir, lowercase):
    def _get_answer_start_and_end(context_list, answer_start, answer_end):
        cur = 0
        answer_from_context = list()
        index_span_list = list()
        for index, word in enumerate(context_list):
            if cur + len(word) - 1 >= answer_start and cur <= answer_end:
                answer_from_context.append(word)
                index_span_list.append(index)
            cur += len(word) + 1
        '''
        if len(answer_from_context) != len(answer_list):
            return None, None

        for index, word in enumerate(answer_list):
            if word != answer_from_context[index]:
                return None, None
        '''
        return index_span_list[0], index_span_list[-1]

    mc_data = []
    with open(train_dir, 'r') as train_file:
        train_data = json.load(train_file)['data']
    paretheses_table = str.maketrans({'(': None, ')': None, ',': None, '.': None, '"': None, '?': None, ';': None, ':': None})
    exception_count = 0
    for dt_pkg in train_data:
        for paragraph in dt_pkg['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                if qa['is_impossible'] is True:
                    continue
                question = qa['question']
                answer = qa['answers'][0]
                answer_text = answer['text']
                answer_start = answer['answer_start']
                answer_end = answer['answer_start'] + len(answer_text) - 1
                context_list = context.split()
                question_list = question.split()
                answer_list = answer_text.split()
                try:
                    start, end = _get_answer_start_and_end(context_list, answer_start, answer_end)
                except:
                    exception_count += 1
                    print('exception_count: ', exception_count)
                for idx, word in enumerate(context_list):
                    context_list[idx] = word.translate(paretheses_table)
                for idx, word in enumerate(question_list):
                    question_list[idx] = word.translate(paretheses_table)
                for idx, word in enumerate(answer_list):
                    answer_list[idx] = word.translate(paretheses_table)
                if context_list[start: end + 1] != answer_list:
                    print(context_list[start: end + 1])
                    print(answer_list)
                else:
                    mc_data.append({
                        'context': context_list,
                        'question': question_list,
                        'answer': [start, end]
                    })
    print('=' * 50)
    print('mc_data_total_size', len(mc_data))
    print('exception_counts', exception_count)
    with open('./data/train.json', 'w') as f:
        f.write(json.dumps(mc_data))

def main():
    parser = argparse.ArgumentParser(description='Preprocess SQUAD v2.0')
    parser.add_argument('--config', default='./config/preprocess_data.config')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = json.loads(config_file.read())
        preprocess_squad_data(
            train_dir=config['train_dir'],
            dev_dir=config['dev_dir'],
            lowercase=False
        )


if __name__ == '__main__':
    main()
