import torch
from torch.utils.data import Dataset


class SquadDataset(Dataset):
    def __init__(self, data, padding_idx, context_max_len, question_max_len):
        selected_data = {
            'context': list(),
            'question': list(),
            'answer': list()
        }
        for idx, context in enumerate(data['context']):
            context_len = min(len(context), context_max_len)
            answer_start, answer_end = data['answer'][idx]
            if answer_start > context_len - 1 or answer_end > context_len - 1:
                continue
            selected_data['context'].append(context)
            selected_data['question'].append(data['question'][idx])
            selected_data['answer'].append(data['answer'][idx])
        self.data_size = len(selected_data['context'])
        self.context_data = torch.ones(self.data_size, context_max_len, dtype=torch.long) * padding_idx
        self.question_data = torch.ones(self.data_size, question_max_len, dtype=torch.long) * padding_idx
        self.context_length = [min(len(context), context_max_len) for context in selected_data['context']]
        self.question_length = [min(len(question), question_max_len) for question in selected_data['question']]
        self.answer = selected_data['answer']
        for idx, context in enumerate(selected_data['context']):
            end_pos = min(len(context), context_max_len)
            self.context_data[idx][:end_pos] = torch.tensor(context[:end_pos], dtype=torch.long)
        for idx, question in enumerate(selected_data['question']):
            end_pos = min(len(question), question_max_len)
            self.question_data[idx][:end_pos] = torch.tensor(question[:end_pos], dtype=torch.long)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return {
            'context': self.context_data[index],
            'question': self.question_data[index],
            'context_length': self.context_length[index],
            'question_length': self.question_length[index],
            'answer': self.answer[index]
        }
