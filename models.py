# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from layers import MLP, EraseAddGate, MLPEncoder, MLPDecoder, ScaledDotProductAttention
from utils import gumbel_softmax

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class GKT(nn.Module):

    def __init__(self, concept_num, hidden_dim, embedding_dim, edge_type_num, graph_type, graph=None, graph_model=None, dropout=0.5, bias=True, binary=False, has_cuda=False):
        super(GKT, self).__init__()
        self.concept_num = concept_num
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.edge_type_num = edge_type_num
        self.device = torch.device('cuda' if has_cuda else 'cpu')
        self.res_len = 2 if binary else 12
        self.has_cuda = has_cuda

        assert graph_type in ['Dense', 'Transition', 'DKT', 'PAM', 'MHA', 'VAE']
        self.graph_type = graph_type
        if graph_type in ['Dense', 'Transition', 'DKT']:
            assert edge_type_num == 2
            assert graph is not None and graph_model is None
            self.graph = nn.Parameter(graph)  # [concept_num, concept_num]
            self.graph.requires_grad = False  # fix parameter
            self.graph_model = graph_model
        else:  # ['PAM', 'MHA', 'VAE']
            assert graph is None
            self.graph = graph  # None
            if graph_type == 'PAM':
                assert graph_model is None
                self.graph = nn.Parameter(torch.rand(concept_num, concept_num))
            else:
                assert graph_model is not None
            self.graph_model = graph_model

    # 수정: one-hot 초기화를 지연 계산 방식으로 변경
        self.one_hot_feat = None  # CPU에서 생성 후 필요 시 초기화
        self.one_hot_q = None

        # 임베딩 레이어
        self.emb_x = nn.Embedding(self.res_len * concept_num, embedding_dim)
        self.emb_c = nn.Embedding(concept_num + 1, embedding_dim, padding_idx=-1)

        # MLP와 게이트
        mlp_input_dim = hidden_dim + embedding_dim
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.f_neighbor_list = nn.ModuleList()
        if graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        else:
            for _ in range(edge_type_num):
                self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        # Erase & Add Gate와 GRU
        self.erase_add_gate = EraseAddGate(hidden_dim, concept_num)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

        # GPU 사용 여부 확인
        print(f"Model initialized on {'CUDA' if has_cuda else 'CPU'}")

    def get_one_hot_feat(self):
        """
        필요 시 CPU에서 one-hot 텐서를 생성하여 반환
        """
        if self.one_hot_feat is None:
            print("Initializing one-hot features on CPU.")
            self.one_hot_feat = torch.eye(self.res_len * self.concept_num, device='cpu')
            self.one_hot_q = torch.cat(
                (torch.eye(self.concept_num, device='cpu'), torch.zeros(1, self.concept_num, device='cpu')), dim=0
            )
        return self.one_hot_feat

    def _aggregate(self, xt, qt, ht, batch_size):
        """
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        
        # 수정: 지연 초기화된 one_hot_feat 사용
        one_hot_feat = self.get_one_hot_feat()
        
        x_idx_mat = torch.arange(self.res_len * self.concept_num, device=self.emb_x.weight.device)
        x_embedding = self.emb_x(x_idx_mat)  # [res_len * concept_num, embedding_dim]
        
        # 수정: masked_feat 생성 시 CPU 사용 후 결과를 GPU로 이동
        masked_feat = F.embedding(xt[qt_mask].cpu(), one_hot_feat)
        res_embedding = masked_feat.mm(x_embedding.cpu()).to(self.device)
        
        concept_idx_mat = self.concept_num * torch.ones((batch_size, self.concept_num), device='cpu').long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.concept_num, device='cpu')
        concept_embedding = self.emb_c(concept_idx_mat.to(self.device))  # [batch_size, concept_num, embedding_dim]
        
        index_tuple = (torch.arange(res_embedding.shape[0], device=self.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        
        # 수정: ht를 GPU로 이동하여 concat 수행
        tmp_ht = torch.cat((ht.to(self.device), concept_embedding), dim=-1)
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            qt: [batch_size]
            m_next: [batch_size, concept_num, hidden_dim]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
        mask_num = masked_tmp_ht.shape[0]

        # Ensure `masked_tmp_ht` has correct dimensions
        if masked_tmp_ht.dim() == 2:
            masked_tmp_ht = masked_tmp_ht.unsqueeze(dim=1)  # Add a dimension to make it [mask_num, 1, hidden_dim + embedding_dim]

        # Ensure dimensions are compatible for repeat
        expanded_self_ht = masked_tmp_ht.unsqueeze(dim=1).repeat(1, self.concept_num, 1).cpu()
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht.cpu()), dim=-1).to(self.device)
        
        self_index_tuple = (torch.arange(mask_num, device=qt.device), qt[qt_mask].long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + embedding_dim]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        
        # Update adjacency operations
        if self.graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            adj = self.graph[qt[qt_mask].long(), :].unsqueeze(dim=-1).cpu()
            neigh_features = adj * self.f_neighbor_list[0](neigh_ht)
        else:  # ['MHA', 'VAE']
            concept_index = torch.arange(self.concept_num, device=qt.device)
            concept_embedding = self.emb_c(concept_index)  # [concept_num, embedding_dim]
            if self.graph_type == 'MHA':
                query = self.emb_c(qt_mask)
                key = concept_embedding
                att_mask = Variable(torch.ones(self.edge_type_num, mask_num, self.concept_num, device=qt.device))
                for k in range(self.edge_type_num):
                    index_tuple = (torch.arange(mask_num, device=qt.device), qt_mask.long())
                    att_mask[k] = att_mask[k].index_put(index_tuple, torch.zeros(mask_num, device=qt.device))
                graphs = self.graph_model(qt_mask, query, key, att_mask)
            else:  # self.graph_type == 'VAE'
                sp_send, sp_rec, sp_send_t, sp_rec_t = self._get_edges(qt_mask)
                graphs, rec_embedding, z_prob = self.graph_model(concept_embedding, sp_send, sp_rec, sp_send_t, sp_rec_t)
            neigh_features = 0
            for k in range(self.edge_type_num):
                adj = graphs[k][qt_mask, :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
                if k == 0:
                    neigh_features = adj * self.f_neighbor_list[k](neigh_ht)
                else:
                    neigh_features = neigh_features + adj * self.f_neighbor_list[k](neigh_ht)
            if self.graph_type == 'MHA':
                neigh_features = 1. / self.edge_type_num * neigh_features
        # neigh_features: [mask_num, concept_num, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next






    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            ht: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht, qt)  # [batch_size, concept_num, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim), ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device), )
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim))
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y

    def _get_next_pred(self, yt, q_next):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = q_next
        next_qt = torch.where(next_qt != -1, next_qt, self.concept_num * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, concept_num]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
        return pred

    # Get edges for edge inference in VAE
    def _get_edges(self, masked_qt):
        r"""
        Parameters:
            masked_qt: qt index with -1 padding values removed
        Shape:
            masked_qt: [mask_num, ]
            rel_send: [edge_num, concept_num]
            rel_rec: [edge_num, concept_num]
        Return:
            rel_send: from nodes in edges which send messages to other nodes
            rel_rec:  to nodes in edges which receive messages from other nodes
        """
        mask_num = masked_qt.shape[0]
        row_arr = masked_qt.cpu().numpy().reshape(-1, 1)  # [mask_num, 1]
        row_arr = np.repeat(row_arr, self.concept_num, axis=1)  # [mask_num, concept_num]
        col_arr = np.arange(self.concept_num).reshape(1, -1)  # [1, concept_num]
        col_arr = np.repeat(col_arr, mask_num, axis=0)  # [mask_num, concept_num]
        # add reversed edges
        new_row = np.vstack((row_arr, col_arr))  # [2 * mask_num, concept_num]
        new_col = np.vstack((col_arr, row_arr))  # [2 * mask_num, concept_num]
        row_arr = new_row.flatten()  # [2 * mask_num * concept_num, ]
        col_arr = new_col.flatten()  # [2 * mask_num * concept_num, ]
        data_arr = np.ones(2 * mask_num * self.concept_num)
        init_graph = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(self.concept_num, self.concept_num))
        init_graph.setdiag(0)  # remove self-loop edges
        row_arr, col_arr, _ = sp.find(init_graph)
        row_tensor = torch.from_numpy(row_arr).long()
        col_tensor = torch.from_numpy(col_arr).long()
        one_hot_table = torch.eye(self.concept_num, self.concept_num)
        rel_send = F.embedding(row_tensor, one_hot_table)  # [edge_num, concept_num]
        rel_rec = F.embedding(col_tensor, one_hot_table)  # [edge_num, concept_num]
        sp_rec, sp_send = rel_rec.to_sparse(), rel_send.to_sparse()
        sp_rec_t, sp_send_t = rel_rec.T.to_sparse(), rel_send.T.to_sparse()
        sp_send = sp_send.to(device=masked_qt.device)
        sp_rec = sp_rec.to(device=masked_qt.device)
        sp_send_t = sp_send_t.to(device=masked_qt.device)
        sp_rec_t = sp_rec_t.to(device=masked_qt.device)
        return sp_send, sp_rec, sp_send_t, sp_rec_t

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=features.device))
        pred_list = []
        ec_list = []  # concept embedding list in VAE
        rec_list = []  # reconstructed embedding list in VAE
        z_prob_list = []  # probability distribution of latent variable z in VAE
        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)  # [batch_size, concept_num, hidden_dim + embedding_dim]
            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht, qt)  # [batch_size, concept_num, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, concept_num]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
            ec_list.append(concept_embedding)
            rec_list.append(rec_embedding)
            z_prob_list.append(z_prob)
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
        return pred_res, ec_list, rec_list, z_prob_list


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
    """

    def __init__(self, n_head, concept_num, input_dim, d_k, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.concept_num = concept_num
        self.d_k = d_k
        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(n_head, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, attn_score, qt):
        """
        Parameters:
            attn_score: attention score of all queries
            qt: masked question index
        Return:
            graphs: n_head types of inferred graphs
        """
        # 수정: 그래프를 CPU에서 생성하고 float16으로 저장
        graphs = torch.zeros(self.n_head, self.concept_num, self.concept_num, device='cpu', dtype=torch.float16)

        for k in range(self.n_head):
            attn_score_k = attn_score[k].detach().cpu().to(dtype=torch.float16)
            index_tuple = (qt.long(), )
            graphs[k] = graphs[k].index_put(index_tuple, attn_score_k)
            
            # 수정: self.graphs도 CPU에서 관리
            self.graphs.data[k] = self.graphs.data[k].to(dtype=torch.float16)
            self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, attn_score_k)
        
        # GPU로 이동
        return graphs.to(attn_score.device)


    def forward(self, features, questions):
        """
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
        """
        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device='cpu'))  # 수정: 초기화 시 CPU 사용
        pred_list = []

        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            
            # 수정: CPU에서 tmp_ht 계산 후 GPU로 이동
            tmp_ht = self._aggregate(xt.cpu(), qt.cpu(), ht.cpu(), batch_size).to(self.device)
            h_next, _, _, _ = self._update(tmp_ht, ht.to(self.device), qt)
            ht = h_next.to('cpu')  # 수정: 연산 후 다시 CPU로 이동

            yt = self._predict(h_next, qt)
            if i < seq_len - 1:
                pred_list.append(self._get_next_pred(yt, questions[:, i + 1]))

        return torch.stack(pred_list, dim=1)


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, msg_hidden_dim, msg_output_dim, concept_num, edge_type_num=2,
                 tau=0.1, factor=True, dropout=0., bias=True):
        super(VAE, self).__init__()
        self.edge_type_num = edge_type_num
        self.concept_num = concept_num
        self.tau = tau
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim, factor=factor, dropout=dropout, bias=bias)
        self.decoder = MLPDecoder(input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=dropout, bias=bias)
        # inferred latent graph, used for saving and visualization
        self.graphs = nn.Parameter(torch.zeros(edge_type_num, concept_num, concept_num))
        self.graphs.requires_grad = False

    def _get_graph(self, edges, sp_rec, sp_send):
        r"""
        Parameters:
            edges: sampled latent graph edge weights from the probability distribution of the latent variable z
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send: one-hot encoded send-node index(sparse tensor)
        Shape:
            edges: [edge_num, edge_type_num]
            sp_rec: [edge_num, concept_num]
            sp_send: [edge_num, concept_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
        """
        x_index = sp_send._indices()[1].long()  # send node index: [edge_num, ]
        y_index = sp_rec._indices()[1].long()   # receive node index [edge_num, ]
        graphs = Variable(torch.zeros(self.edge_type_num, self.concept_num, self.concept_num, device=edges.device))
        for k in range(self.edge_type_num):
            index_tuple = (x_index, y_index)
            graphs[k] = graphs[k].index_put(index_tuple, edges[:, k])  # used for calculation
            #############################
            # here, we need to detach edges when storing it into self.graphs in case memory leak!
            self.graphs.data[k] = self.graphs.data[k].index_put(index_tuple, edges[:, k].detach())  # used for saving and visualization
            #############################
        return graphs

    def forward(self, data, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            data: input concept embedding matrix
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            data: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            graphs: latent graph list modeled by z which has different edge types
            output: the reconstructed data
            prob: q(z|x) distribution
        """
        logits = self.encoder(data, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [edge_num, output_dim(edge_type_num)]
        edges = gumbel_softmax(logits, tau=self.tau, dim=-1)  # [edge_num, edge_type_num]
        prob = F.softmax(logits, dim=-1)
        output = self.decoder(data, edges, sp_send, sp_rec, sp_send_t, sp_rec_t)  # [concept_num, embedding_dim]
        graphs = self._get_graph(edges, sp_send, sp_rec)
        return graphs, output, prob


class DKT(nn.Module):

    def __init__(self, feature_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(DKT, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        self.rnn = nn.LSTM(feature_dim, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
        self.f_out = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, (nn.LSTM)):
                for i, weight in enumerate(m.parameters()):
                    if i < 2:
                        nn.init.orthogonal_(weight)

    def _get_next_pred(self, yt, questions):
        r"""
        Parameters:
            y: predicted correct probability of all concepts at the next timestamp
            questions: question index matrix
        Shape:
            y: [batch_size, seq_len - 1, output_dim]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        one_hot = torch.eye(self.output_dim, device=yt.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim, device=yt.device)), dim=0)
        next_qt = questions[:, 1:]
        next_qt = torch.where(next_qt != -1, next_qt, self.output_dim * torch.ones_like(next_qt, device=yt.device))  # [batch_size, seq_len - 1]
        one_hot_qt = F.embedding(next_qt, one_hot)  # [batch_size, seq_len - 1, output_dim]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=-1)  # [batch_size, seq_len - 1]
        return pred

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        feat_one_hot = torch.eye(self.feature_dim, device=features.device)
        feat_one_hot = torch.cat((feat_one_hot, torch.zeros(1, self.feature_dim, device=features.device)), dim=0)
        feat = torch.where(features != -1, features, self.feature_dim * torch.ones_like(features, device=features.device))
        features = F.embedding(feat, feat_one_hot)

        feature_lens = torch.ne(questions, -1).sum(dim=1)  # padding value = -1
        x_packed = pack_padded_sequence(features, feature_lens, batch_first=True, enforce_sorted=False)
        output_packed, _ = self.rnn(x_packed)  # [batch, seq_len, hidden_dim]
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)  # [batch, seq_len, hidden_dim]
        yt = self.f_out(output_padded)  # [batch, seq_len, output_dim]
        yt = torch.sigmoid(yt)
        yt = yt[:, :-1, :]  # [batch, seq_len - 1, output_dim]
        pred_res = self._get_next_pred(yt, questions)  # [batch, seq_len - 1]
        return pred_res
