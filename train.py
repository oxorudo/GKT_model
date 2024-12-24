import numpy as np
import time
import random
import argparse
import pickle
import os
import gc
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models import GKT, MultiHeadAttention, VAE, DKT
from metrics import KTLoss, VAELoss
from processing import load_dataset
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
import requests
from mlflow.tracking import MlflowClient

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

# .env 파일 로드
load_dotenv(dotenv_path=".env")

# 환경 변수 설정
MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# 디버깅: 환경 변수 출력
print("MLFLOW_SERVER_URI:", MLFLOW_SERVER_URI)
print("SLACK_WEBHOOK_URL:", SLACK_WEBHOOK_URL)

EXPERIMENT_NAME = "GKT testing_plus"
MODEL_NAME = "GKT+"
ACCURACY_THRESHOLD = 0.7  # 성능 검증 기준값


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_false', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data-dir', type=str, default='data', help='Data dir for loading input data.')
parser.add_argument('--data-file', type=str, default='filtered_combined_user_data.csv', help='Name of input data file.')
parser.add_argument('--save-dir', type=str, default='logs', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('-graph-save-dir', type=str, default='graphs', help='Dir for saving concept graphs.')
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--dkt-graph-dir', type=str, default='dkt-graph', help='Where to load the pretrained dkt graph.')
parser.add_argument('--dkt-graph', type=str, default='dkt_graph.txt', help='DKT graph data file name.')
parser.add_argument('--model', type=str, default='GKT', help='Model type to use, support GKT and DKT.')
parser.add_argument('--hid-dim', type=int, default=32, help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=32, help='Dimension of concept embedding.')
parser.add_argument('--attn-dim', type=int, default=32, help='Dimension of multi-head attention layers.')
parser.add_argument('--vae-encoder-dim', type=int, default=32, help='Dimension of hidden layers in vae encoder.')
parser.add_argument('--vae-decoder-dim', type=int, default=32, help='Dimension of hidden layers in vae decoder.')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--graph-type', type=str, default='Dense', help='The type of latent concept graph.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
parser.add_argument('--binary', type=bool, default=True, help='Whether only use 0/1 for results.')
parser.add_argument('--result-type', type=int, default=12, help='Number of results types when multiple results are used.')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
parser.add_argument('--var', type=float, default=1, help='Output variance.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.6, help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.2, help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
parser.add_argument('--test-model-dir', type=str, default='logs/expDKT', help='Existed model file dir.')
parser.add_argument('--max-users', type=int, default=None, help='유저수 제한')
parser.add_argument('--max-seq', type=int, default=None, help='최근 풀이 시퀀스 길이 제한')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# CUDA가 사용 가능한지 확인
cuda_available = torch.cuda.is_available()
print(f"CUDA 사용 가능 여부: {cuda_available}")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

res_len = 2 if args.binary else args.result_type

# Save model and meta-data. Always saves in a new sub-folder.
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")

# load dataset
dataset_path = os.path.join(args.data_dir, args.data_file)
dkt_graph_path = os.path.join(args.dkt_graph_dir, args.dkt_graph)
if not os.path.exists(dkt_graph_path):
    dkt_graph_path = None
concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(dataset_path, args.batch_size, args.graph_type, max_users=args.max_users, max_seq = args.max_seq, dkt_graph_path=dkt_graph_path,
                                                                           train_ratio=args.train_ratio, val_ratio=args.val_ratio, shuffle=args.shuffle,
                                                                           model_type=args.model, use_cuda=args.cuda)

# build models
graph_model = None
if args.model == 'GKT':
    if args.graph_type == 'MHA':
        graph_model = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim, args.attn_dim, dropout=args.dropout)
    elif args.graph_type == 'VAE':
        graph_model = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types, args.vae_decoder_dim, args.vae_decoder_dim, concept_num,
                          edge_type_num=args.edge_types, tau=args.temp, factor=args.factor, dropout=args.dropout, bias=args.bias)
        vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior, var=args.var)
        if args.cuda:
            vae_loss = vae_loss.cuda()
    if args.cuda and args.graph_type in ['MHA', 'VAE']:
        graph_model = graph_model.cuda()
    model = GKT(concept_num, args.hid_dim, args.emb_dim, args.edge_types, args.graph_type, graph=graph, graph_model=graph_model,
                dropout=args.dropout, bias=args.bias, has_cuda=args.cuda)
elif args.model == 'DKT':
    model = DKT(res_len * concept_num, args.emb_dim, concept_num, dropout=args.dropout, bias=args.bias)
else:
    raise NotImplementedError(args.model + ' model is not implemented!')
kt_loss = KTLoss()

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=args.gamma, patience=5, verbose=True, min_lr=1e-6
)

# load model/optimizer/scheduler params
if args.load_dir:
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    model_file = os.path.join(args.load_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    scheduler.load_state_dict(torch.load(scheduler_file))
    args.save_dir = False

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.model == 'GKT' and args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    model = model.cuda()
    kt_loss = KTLoss()

# 에포크마다 그래프를 저장하는 코드 추가
def save_graph(epoch, graph_model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    graph_file = os.path.join(save_dir, f'graph_epoch.pt')
    torch.save(graph_model.state_dict(), graph_file)
    print(f"Graph at epoch {epoch} saved to {graph_file}")

def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    kt_train = []
    vae_train = []
    auc_train = []
    acc_train = []
    if graph_model is not None:
        graph_model.train()
    model.train()
    for batch_idx, (features, questions, answers) in enumerate(train_loader):
        t1 = time.time()
        if args.cuda:
            features = features.pin_memory().cuda(non_blocking=True)
            questions = questions.pin_memory().cuda(non_blocking=True)
            answers = answers.pin_memory().cuda(non_blocking=True)
        ec_list, rec_list, z_prob_list = None, None, None
        if args.model == 'GKT':
            pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
        elif args.model == 'DKT':
            pred_res = model(features, questions)
        else:
            raise NotImplementedError(args.model + ' model is not implemented!')
        loss_kt, auc, acc = kt_loss(pred_res, answers)
        kt_train.append(float(loss_kt.cpu().detach().numpy()))
        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)

        if args.model == 'GKT' and args.graph_type == 'VAE':
            if args.prior:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
            else:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                vae_train.append(float(loss_vae.cpu().detach().numpy()))
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'loss vae: ', loss_vae.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
            loss = loss_kt + loss_vae
        else:
            loss = loss_kt
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
        loss_train.append(float(loss.cpu().detach().numpy()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss
        print('cost time: ', str(time.time() - t1))

    loss_val = []
    kt_val = []
    vae_val = []
    auc_val = []
    acc_val = []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(valid_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')
            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt = float(loss_kt.cpu().detach().numpy())
            kt_val.append(loss_kt)
            if auc != -1 and acc != -1:
                auc_val.append(auc)
                acc_val.append(acc)

            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_val.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_val.append(loss)
            del loss
    if args.model == 'GKT' and args.graph_type == 'VAE':
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'kt_train: {:.10f}'.format(np.mean(kt_train)),
              'vae_train: {:.10f}'.format(np.mean(vae_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'kt_val: {:.10f}'.format(np.mean(kt_val)),
              'vae_val: {:.10f}'.format(np.mean(vae_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    if args.save_dir and np.mean(loss_val) < best_val_loss:
        print('Best model so far, saving...')
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        torch.save(scheduler.state_dict(), scheduler_file)

        # 그래프 저장 (최적 모델일 때만)
        if graph_model is not None:
            save_graph(epoch, graph_model, args.graph_save_dir)
        
        if args.model == 'GKT' and args.graph_type == 'VAE':
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'kt_train: {:.10f}'.format(np.mean(kt_train)),
                  'vae_train: {:.10f}'.format(np.mean(vae_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'kt_val: {:.10f}'.format(np.mean(kt_val)),
                  'vae_val: {:.10f}'.format(np.mean(vae_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            del kt_train
            del vae_train
            del kt_val
            del vae_val
        else:
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    res = [np.mean(loss_train), np.mean(auc_train), np.mean(acc_train), np.mean(loss_val), np.mean(auc_val), np.mean(acc_val)]
    scheduler.step(np.mean(loss_val))
    del loss_train
    del auc_train
    del acc_train
    del loss_val
    del auc_val
    del acc_val
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res


def test_with_mlflow():
    with mlflow.start_run(nested=True) as run:
        mlflow.set_tracking_uri(MLFLOW_SERVER_URI) # mlflow uri 설정
        mlflow.set_experiment(EXPERIMENT_NAME)  # 실험 이름 설정

        print('--------------------------------')
        print('--------Testing mlflow----------')
        print('--------------------------------')

        loss_test = []
        kt_test = []
        vae_test = []
        auc_test = []
        acc_test = []

        if graph_model is not None:
            graph_model.eval()
        model.eval()
        model.load_state_dict(torch.load(model_file))

        with torch.no_grad():
            for batch_idx, (features, questions, answers) in enumerate(test_loader):
                if args.cuda:
                    features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()

                # 모델 예측
                ec_list, rec_list, z_prob_list = None, None, None
                if args.model == 'GKT':
                    pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
                elif args.model == 'DKT':
                    pred_res = model(features, questions)
                else:
                    raise NotImplementedError(args.model + ' model is not implemented!')

                # 손실 및 성능 계산
                loss_kt, auc, acc = kt_loss(pred_res, answers)
                loss_kt = float(loss_kt.cpu().detach().numpy())
                if auc != -1 and acc != -1:
                    auc_test.append(auc)
                    acc_test.append(acc)
                kt_test.append(loss_kt)
                loss = loss_kt

                # VAE 손실 계산 (GKT 모델 + VAE 사용 시)
                if args.model == 'GKT' and args.graph_type == 'VAE':
                    loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                    loss_vae = float(loss_vae.cpu().detach().numpy())
                    vae_test.append(loss_vae)
                    loss += loss_vae
                loss_test.append(loss)

        # 평균 손실 및 성능 계산
        mean_loss_test = np.mean(loss_test)
        mean_kt_test = np.mean(kt_test)
        mean_auc_test = np.mean(auc_test)
        mean_acc_test = np.mean(acc_test)
        mean_vae_test = np.mean(vae_test) if vae_test else None

        # MLflow에 결과 기록
        mlflow.log_metric("loss_test", mean_loss_test)
        mlflow.log_metric("auc_test", mean_auc_test)
        mlflow.log_metric("acc_test", mean_acc_test)
        if mean_vae_test:
            mlflow.log_metric("vae_test", mean_vae_test)
            mlflow.log_metric("kt_test", mean_kt_test)

        # 결과 출력
        print(f"loss_test: {mean_loss_test:.10f}",
              f"auc_test: {mean_auc_test:.10f}",
              f"acc_test: {mean_acc_test:.10f}")
        if mean_vae_test:
            print(f"vae_test: {mean_vae_test:.10f}")
            print(f"kt_test: {mean_kt_test:.10f}")

        # Slack 알림
        message = f"테스트가 완료되었습니다.\nLoss: {mean_loss_test:.10f}, AUC: {mean_auc_test:.10f}, Accuracy: {mean_acc_test:.10f}"
        if mean_vae_test:
            message += f", VAE Loss: {mean_vae_test:.10f}"
        send_slack_notification("테스트 완료", message)

        # 모델 등록
        artifact_uri = mlflow.get_artifact_uri("best_model")
        register_model(run.info.run_id, artifact_uri, mean_acc_test)
        mlflow.pytorch.log_model(model, "best_model")

        # 로그 파일 기록
        if args.save_dir:
            print('--------------------------------', file=log)
            print('--------Testing-----------------', file=log)
            print('--------------------------------', file=log)
            print(f"loss_test: {mean_loss_test:.10f}",
                  f"kt_test: {mean_kt_test:.10f}",
                  f"auc_test: {mean_auc_test:.10f}",
                  f"acc_test: {mean_acc_test:.10f}", file=log)
            if mean_vae_test:
                print(f"vae_test: {mean_vae_test:.10f}", file=log)
            log.flush()

        # 자원 정리
        del loss_test, kt_test, vae_test, auc_test, acc_test
        gc.collect()
        if args.cuda:
            torch.cuda.empty_cache()

# Slack 알림 함수 정의
def send_slack_notification(status, message):
    """Slack 알림 전송"""
    if not SLACK_WEBHOOK_URL:
        print("Slack Webhook URL이 설정되지 않았습니다.")
        return

    payload = {"text": f"MLflow 작업 상태: {status}\n{message}"}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print("Slack 알림 성공")
    except requests.exceptions.RequestException as e:
        print(f"Slack 알림 실패: {str(e)}")

# 모델 등록 함수 정의
def register_model(run_id, artifact_uri, accuracy):
    """MLflow 모델 레지스트리에 등록"""
    MODEL_NAME = "GKT+"  # 모델 이름 설정
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        print(f"Model {MODEL_NAME} already exists. Skipping creation.")

    # 모델 버전 생성
    try:
        model_version = client.create_model_version(
            name=MODEL_NAME, source=artifact_uri, run_id=run_id
        )
        print(f"Model version {model_version.version} created.")
        send_slack_notification(
            status="성공",
            message=f"모델 등록 성공\nModel: {MODEL_NAME}\nVersion: {model_version.version}",
        )

        # 성능 기준에 따라 모델 단계 전환
        target_stage = "Production" if accuracy >= ACCURACY_THRESHOLD else "Staging"
        client.transition_model_version_stage(
            name=MODEL_NAME, version=model_version.version, stage=target_stage
        )
        print(f"Model version {model_version.version} moved to {target_stage}.")
        send_slack_notification(
            status="성공",
            message=f"모델 {target_stage} 단계로 전환 완료\nModel: {MODEL_NAME}\nVersion: {model_version.version}",
        )
    except Exception as e:
        send_slack_notification(status="실패", message=f"모델 등록 중 오류 발생: {str(e)}")
        raise


mlflow.set_tracking_uri(MLFLOW_SERVER_URI) # mlflow uri 설정
mlflow.set_experiment(EXPERIMENT_NAME)  # 실험 이름 설정


if args.test is False:

    with mlflow.start_run(nested=True) as run:
        # MLflow에 파라미터 기록
        mlflow.log_params(vars(args))

        print('start training!')
        t_total = time.time()
        best_val_loss = np.inf
        best_epoch = 0
        for epoch in range(args.epochs):
            list = train(epoch, best_val_loss)
            
            # 에포크 결과를 MLflow에 기록
            mlflow.log_metric("loss_train", list[0], step=epoch)
            mlflow.log_metric("auc_train", list[1], step=epoch)
            mlflow.log_metric("acc_train", list[2], step=epoch)
            mlflow.log_metric("loss_val", list[3], step=epoch)
            mlflow.log_metric("auc_val", list[4], step=epoch)
            mlflow.log_metric("acc_val", list[5], step=epoch)


            if list[3] < best_val_loss:
                best_val_loss = list[3]
                best_epoch = epoch

        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))

        # 최종 정보 기록
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)

        # Slack 알림: 학습 완료
        send_slack_notification("완료", f"최적의 모델이 저장되었습니다.\nBest Epoch: {best_epoch}, Best Validation Loss: {best_val_loss:.10f}")

        if args.save_dir:
            print("Best Epoch: {:04d}".format(best_epoch), file=log)
            log.flush()

        test_with_mlflow()

