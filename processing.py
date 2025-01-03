import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph
import os

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

class KTDataset(Dataset):
    def __init__(self, features, questions, answers):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    (features, questions, answers) = zip(*batch)
    features = [torch.LongTensor(feat).to(torch.device('cuda', index=0)) for feat in features]
    questions = [torch.LongTensor(qt).to(torch.device('cuda', index=0)) for qt in questions]
    answers = [torch.LongTensor(ans).to(torch.device('cuda', index=0)) for ans in answers]

    # 패딩 처리
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)

    # 데이터 위치 확인
    print("Batch Data Locations:")
    print(f"  Features: {feature_pad.device}, Questions: {question_pad.device}, Answers: {answer_pad.device}")
    return feature_pad, question_pad, answer_pad


def load_dataset(file_path, batch_size, graph_type, max_users, max_seq, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    df = pd.read_csv(file_path)
    if "f_mchapter_id" not in df.columns:
        raise KeyError(f"The column 'f_mchapter_id' was not found on {file_path}")
    if "Correct" not in df.columns:
        raise KeyError(f"The column 'Correct' was not found on {file_path}")
    if "UserID" not in df.columns:
        raise KeyError(f"The column 'UserID' was not found on {file_path}")

    # if not (df['correct'].isin([0, 1])).all():
    #     raise KeyError(f"The values of the column 'correct' must be 0 or 1.")
    # df = df.iloc[0:100000,:]

    # Step 0 - 정렬: 가장 오래된 기록부터 정렬
    df.sort_values(by=["UserID", "CreDate"], inplace=True)  # "CreDate" 컬럼을 기준으로 정렬

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['f_mchapter_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('UserID').filter(lambda q: len(q) > 1).copy()

    # Step 1.3 - 유저당 최근 1000개의 풀이 데이터만 유지
    if max_seq is not None:
        df = df.groupby('UserID').tail(max_seq).copy()


    # Step 1 - 유저 제한
    if max_users is not None:
        unique_users = df['UserID'].unique()
        if len(unique_users) > max_users:
            selected_users = np.random.choice(unique_users, max_users, replace=False)
            df = df[df['UserID'].isin(selected_users)]


    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['f_mchapter_id'], sort=True)  # we can also use problem_id to represent exercises

    # correct 생성 (O -> 1, X -> 0)
    df['Correct'] = df['Correct'].map({'O': 1, 'X': 0})

    # Step 3 - Cross skill id with answer to form a synthetic feature
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    if use_binary:
        df['skill_with_answer'] = df['skill'] * 2 + df['Correct']
    else:
        df['skill_with_answer'] = df['skill'] * res_len + df['Correct'] - 1


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []

    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['Correct'].eq(1).astype('int').tolist())
        seq_len_list.append(series['Correct'].shape[0])

    df.groupby('UserID').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(df['skill_with_answer'].max() + 1)
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = question_dim

    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        if use_cuda and graph_type in ['Dense', 'Transition', 'DKT']:
            graph = graph.cuda()
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader




def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph