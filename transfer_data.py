import argparse
import json


def transfer_squad_data(input_dir, output_dir, lowercase):
    def _get_answer_start_and_end(context_list, answer_start, answer_end):
        cur = 0
        answer_from_context = list()
        index_span_list = list()
        for index, word in enumerate(context_list):
            if cur + len(word) - 1 >= answer_start and cur <= answer_end:
                answer_from_context.append(word)
                index_span_list.append(index)
            cur += len(word) + 1
        return index_span_list[0], index_span_list[-1]

    mc_data = []
    with open(input_dir, 'r') as train_file:
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
                    if lowercase is True:
                        context_list[idx] = context_list[idx].lower()
                for idx, word in enumerate(question_list):
                    question_list[idx] = word.translate(paretheses_table)
                    if lowercase is True:
                        question_list[idx] = question_list[idx].lower()
                for idx, word in enumerate(answer_list):
                    answer_list[idx] = word.translate(paretheses_table)
                    if lowercase is True:
                        answer_list[idx] = answer_list[idx].lower()
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
    print('=' * 50)
    with open(output_dir, 'w') as f:
        f.write(json.dumps(mc_data))


def main():
    parser = argparse.ArgumentParser(description='Transfer SQUAD v2.0')
    parser.add_argument('--config', default='./config/transfer_data.config')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config = json.loads(config_file.read())
        transfer_squad_data(
            input_dir=config['input_train_dir'],
            output_dir=config['output_train_dir'],
            lowercase=True
        )
        transfer_squad_data(
            input_dir=config['input_dev_dir'],
            output_dir=config['output_dev_dir'],
            lowercase=True
        )


if __name__ == '__main__':
    main()
