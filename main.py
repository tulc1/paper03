import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor, cus_write_txt
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData

device = 'cuda:' + args.cuda

# hyperparameters
d = args.d
l = args.gnn_layer
hyper = args.hyper
temp_1 = args.temp1
temp_2 = args.temp2
temp_3 = args.temp3
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
lambda_3 = args.lambda3
reg = args.reg
dropout = args.dropout
lr = args.lr
decay = args.decay

current_t = time.strftime('%Y-%m-%d-%H-%M', time.gmtime())
cus_write_txt(str(args), args, current_t)

torch.manual_seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)

# load data
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
print('Data loaded.')

print('dataset:', args.data, " layer:", l)
print('user_num:', train.shape[0], 'item_num:', train.shape[1], 'lambda_1:', lambda_1, 'lambda_2:', lambda_2, 'lambda_3:', lambda_3,
      'temp1:', temp_1, 'temp2:', temp_2, 'temp3:', temp_3, 'reg:', reg, 'hyper:', hyper)

epoch_user = min(train.shape[0], 30000)

tmp_train = train.copy()
tmp_adj = scipy_sparse_mat_to_torch_sparse_tensor(tmp_train.tocoo())
tmp_adj = tmp_adj.coalesce().cuda(torch.device(device))

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = LightGCL(args, adj_norm.shape[0], adj_norm.shape[1], d, hyper, train_csr, tmp_adj, adj_norm, dropout, batch_user, device)
#model.load_state_dict(torch.load('saved_model.pt'))
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model and optimizer initialized. Total params:', params)

current_lr = lr

for epoch in range(epoch_no):
    # if (epoch+1)%50 == 0:
    #     torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
    #     torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')

    epoch_loss, epoch_loss_r, epoch_loss_cl, epoch_loss_reg = 0, 0, 0, 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_cl, loss_reg = model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_cl += loss_cl.cpu().item()
        epoch_loss_reg += loss_reg.cpu().item()

        torch.cuda.empty_cache()

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_cl = epoch_loss_cl/batch_no
    epoch_loss_reg = epoch_loss_reg/batch_no

    print('Epoch:', epoch, 'Loss:', epoch_loss, 'Loss_r:', epoch_loss_r, 'Loss_cl:', epoch_loss_cl,
          'Loss_reg', epoch_loss_reg)

    if epoch == 24:
        user_emb = model.E_u
        item_emb = model.E_i
        np.savetxt("./emb/" + args.data + "_user_emb.csv", user_emb.cpu().detach().numpy(), delimiter=",")
        np.savetxt("./emb/" + args.data + "_item_emb.csv", item_emb.cpu().detach().numpy(), delimiter=",")
        exit()

    if epoch % 4 == 0:
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))

        all_recall_20, all_ndcg_20, all_recall_40, all_ndcg_40 = 0, 0, 0, 0
        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
            predictions = model(test_uids_input, None, None, None, test=True)
            predictions = np.array(predictions.cpu())

            # top@20
            recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
            # top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

            all_recall_20 += recall_20
            all_ndcg_20 += ndcg_20
            all_recall_40 += recall_40
            all_ndcg_40 += ndcg_40
        print('-------------------------------------------')
        print('Test of epoch', epoch, ':', 'Recall@20:', all_recall_20/batch_no, 'Ndcg@20:', all_ndcg_20/batch_no,
              'Recall@40:', all_recall_40/batch_no, 'Ndcg@40:', all_ndcg_40/batch_no)
        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20/batch_no)
        ndcg_20_y.append(all_ndcg_20/batch_no)
        recall_40_y.append(all_recall_40/batch_no)
        ndcg_40_y.append(all_ndcg_40/batch_no)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20, all_ndcg_20, all_recall_40, all_ndcg_40 = 0, 0, 0, 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
    predictions = model(test_uids_input,None,None,None,test=True)
    predictions = np.array(predictions.cpu())

    # top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    # top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20 += recall_20
    all_ndcg_20 += ndcg_20
    all_recall_40 += recall_40
    all_ndcg_40 += ndcg_40
print('-------------------------------------------')
print('Final test:', 'Recall@20:', all_recall_20/batch_no, 'Ndcg@20:', all_ndcg_20/batch_no,
      'Recall@40:', all_recall_40/batch_no, 'Ndcg@40:', all_ndcg_40/batch_no)

recall_20_x.append('Final')
recall_20_y.append(all_recall_20/batch_no)
ndcg_20_y.append(all_ndcg_20/batch_no)
recall_40_y.append(all_recall_40/batch_no)
ndcg_40_y.append(all_ndcg_40/batch_no)

metric = pd.DataFrame({
    'epoch': recall_20_x,
    'recall@20': recall_20_y,
    'ndcg@20': ndcg_20_y,
    'recall@40': recall_40_y,
    'ndcg@40': ndcg_40_y
})

metric.to_csv('log/result_'+args.data+'_'+current_t+'.csv')

# torch.save(model.state_dict(),'saved_model/saved_model_'+args.data+'_'+current_t+'.pt')
# torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+args.data+'_'+current_t+'.pt')