import torch
import torch.nn as nn


class Similarity(nn.Module):
    def __init__(self, w_dim):
        super(Similarity, self).__init__()
        self._linear = nn.Linear(w_dim, 1)

    def forward(self, context, context_length, question, question_length):
        batch_size = context.shape[0]
        context_max_len = context.shape[1]
        question_max_len = question.shape[1]
        similarity_matrix = list()
        for idx in range(context_max_len):
            batch_context_word = context[:, idx, :].unsqueeze(1).expand(batch_size, question_max_len, -1)
            x = torch.cat([batch_context_word, question, batch_context_word*question], dim=-1)
            word_sim = self._linear(x).transpose(1, 2)
            # print('word_sim.size(): ', word_sim.size())
            similarity_matrix.append(word_sim)
        similarity = torch.cat(similarity_matrix, dim=1)
        # print('similarity.size(): ', similarity.size())
        return similarity


class Context2Query(nn.Module):
    def __init__(self):
        super(Context2Query, self).__init__()

    def forward(self, similarity, question):
        context_query = similarity.bmm(question)
        return context_query


class Query2Context(nn.Module):
    def __init__(self):
        super(Query2Context, self).__init__()

    def forward(self, similarity, context, context_query):
        similarity, _ = torch.max(similarity, dim=-1, keepdim=True)
        similarity = nn.functional.softmax(similarity, dim=1)
        similarity = similarity.transpose(1, 2)
        query_context = similarity.bmm(context).expand(-1, context.shape[1], -1)
        G = torch.cat([context, context_query, context*context_query, context*query_context], dim=-1)
        return G


class OutputLayer(nn.Module):
    def __init__(self, w_dim):
        super(OutputLayer, self).__init__()
        self._linear = nn.Linear(w_dim, 1)

    def forward(self, x, context_length):
        x = self._linear(x).squeeze(-1)
        x = nn.functional.softmax(x, dim=-1)
        return x
