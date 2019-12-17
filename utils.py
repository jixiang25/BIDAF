import torch


def tensor_answer(answer, context_max_len):
    batch_size = answer[0].shape[0]
    answer_start = torch.zeros((batch_size, context_max_len), dtype=torch.long)
    answer_end = torch.zeros((batch_size, context_max_len), dtype=torch.long)
    indices = [idx for idx in range(batch_size)]
    answer_start[indices, answer[0]] = 1
    answer_end[indices, answer[1]] = 1
    return answer_start, answer_end


def predict(prob_start, prob_end, context_length):
    predict_answer = []
    batch_size = prob_start.shape[0]
    for batch_idx in range(batch_size):
        length = context_length[batch_idx]
        pos_start, pos_end = -1, -1
        max_probability = 0.0
        for i in range(length):
            for j in range(i + 1, length):
                probability_ij = (prob_start[batch_idx][i] * prob_end[batch_idx][j]).item()
                if probability_ij > max_probability:
                    pos_start, pos_end = i, j
                    max_probability = probability_ij
        predict_answer.append([pos_start, pos_end])
    return predict_answer


def exact_match(predict_answer, answer):
    batch_size = answer[0].shape[0]
    em_count = 0
    for batch_idx in range(batch_size):
        if predict_answer[batch_idx][0] == answer[0][batch_idx].item() and \
        predict_answer[batch_idx][1] == answer[1][batch_idx].item():
            em_count += 1
    return em_count


def f1(predict_answer, answer, context):
    batch_size = answer[0].shape[0]
    for batch_idx in range(batch_size):
        pred_start, pred_end = predict_answer[batch_idx]
        real_start, real_end = answer[0][batch_idx].item(), answer[1][batch_idx].item()
        pred_dict, real_dict = dict(), dict()
        TT, TF, FT = 0, 0, 0
        for i in range(pred_start, pred_end + 1):
            pred_dict[context[batch_idx][i].item()] = 1
        for i in range(real_start, real_end + 1):
            real_dict[context[batch_idx][i].item()] = 1
        for i in range(pred_start, pred_end + 1):
            word = context[batch_idx][i].item()
            if word in real_dict:
                TT += 1
            else:
                FT += 1
        for i in range(real_start, real_end + 1):
            word = context[batch_idx][i].item()
            if word not in pred_dict:
                TF += 1
        precise = TT / (TT + FT)
        recall = TT / (TT + TF)
        f1_value = 0.0 if precise + recall == 0 else 2.0 * precise * recall / (precise + recall)
        return f1_value


def sort_seq_by_len(batch, sequence_length):
    sorted_seq_len, sorting_idx = sequence_length.sort(dim=0, descending=True)
    sorted_batch = batch.index_select(0, sorting_idx)

    _, reverse_mapping = sorting_idx.sort(0, descending=False)
    return sorted_batch, sorted_seq_len, reverse_mapping
