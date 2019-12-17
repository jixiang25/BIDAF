import torch
import torch.nn as nn
from layers import Similarity, Context2Query, Query2Context, OutputLayer
from utils import sort_seq_by_len


class BIDAF(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 embedding,
                 dropout,
                 device,
                 padding_idx):
        super(BIDAF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = embedding
        self.device = device

        # Word Embedding Layer
        self._embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self._embedding.weight = nn.Parameter(embedding, requires_grad=False)

        # Contextual Embedding Layer
        self._contextual_encoding = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Attention Flow Layer
        self._softmax_similarity = Similarity(w_dim=6*embedding_dim)
        self._context2query = Context2Query()
        self._query2context = Query2Context()

        # Modeling Layer
        self._modeling_rnn = nn.LSTM(
            input_size=8*embedding_dim,
            hidden_size=embedding_dim,
            dropout=dropout,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Output Layer
        self._output_start_layer = OutputLayer(w_dim=10*embedding_dim)
        self._m2_rnn = nn.LSTM(
            input_size=2*embedding_dim,
            hidden_size=embedding_dim,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        self._output_end_layer = OutputLayer(w_dim=10*embedding_dim)

    def forward(self, context, context_length, question, question_length):
        embed_context = self._embedding(context)
        embed_question = self._embedding(question)

        sorted_context, sorted_context_len, context_restore = sort_seq_by_len(embed_context, context_length)
        sorted_question, sorted_question_len, question_restore = sort_seq_by_len(embed_question, question_length)
        packed_embed_context = nn.utils.rnn.pack_padded_sequence(sorted_context, sorted_context_len, batch_first=True)
        packed_embed_question = nn.utils.rnn.pack_padded_sequence(sorted_question, sorted_question_len, batch_first=True)
        pad_contextual_context, _ = self._contextual_encoding(packed_embed_context, None)
        pad_contextual_question, _ = self._contextual_encoding(packed_embed_question, None)
        contextual_context, _ = nn.utils.rnn.pad_packed_sequence(pad_contextual_context, batch_first=True)
        contextual_question, _ = nn.utils.rnn.pad_packed_sequence(pad_contextual_question, batch_first=True)
        contextual_context = contextual_context.index_select(0, context_restore)
        contextual_question = contextual_question.index_select(0, question_restore)

        similarity = self._softmax_similarity(context=contextual_context,
                                              context_length=context_length,
                                              question=contextual_question,
                                              question_length=question_length)
        context_query = self._context2query(similarity, contextual_question)
        G = self._query2context(similarity, contextual_context, context_query)

        sorted_M, sorted_M_len, M_restore = sort_seq_by_len(G, context_length)
        packed_M = nn.utils.rnn.pack_padded_sequence(sorted_M, sorted_M_len, batch_first=True)
        pad_M, _ = self._modeling_rnn(packed_M, None)
        M, _ = nn.utils.rnn.pad_packed_sequence(pad_M, batch_first=True)
        M = M.index_select(0, M_restore)

        G_M = torch.cat((G, M), dim=-1)
        probability_start = self._output_start_layer(G_M, context_length)
        sorted_M2, sorted_M2_len, M2_restore = sort_seq_by_len(M, context_length)
        packed_M2 = nn.utils.rnn.pack_padded_sequence(sorted_M2, sorted_M2_len, batch_first=True)
        pad_M2, _ = self._m2_rnn(packed_M2, None)
        M2, _ = nn.utils.rnn.pad_packed_sequence(pad_M2, batch_first=True)
        M2 = M2.index_select(0, M2_restore)
        G_M2 = torch.cat((G, M2), dim=-1)
        probability_end = self._output_end_layer(G_M2, context_length)
        return probability_start, probability_end
