import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCL(nn.Module):
    def __init__(self, args, n_u, n_i, d, h_n, train_csr, adj, adj_norm, dropout, batch_user, device):
        super(LightGCL, self).__init__()
        self.args = args
        self.dim = d
        self.user, self.item = n_u, n_i

        # hyper parameter
        self.layer = args.gnn_layer
        self.h_layer = args.hgnn_layer
        self.h_num = h_n

        self.alpha = args.alpha
        self.temp_1 = args.temp1
        self.temp_2 = args.temp2
        self.temp_3 = args.temp3
        self.lambda_1 = args.lambda1
        self.lambda_2 = args.lambda2
        self.lambda_3 = args.lambda3
        self.reg = args.reg
        self.dropout = dropout
        self.batch_user = batch_user
        self.device = device

        # node embedding
        self.u_embeds = nn.Parameter(init(torch.empty(self.user, self.dim)))
        self.i_embeds = nn.Parameter(init(torch.empty(self.item, self.dim)))

        # batch norm
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(self.dim) for _ in range(self.layer)])
        self.h_bn_layers = nn.ModuleList([nn.BatchNorm1d(self.h_num) for _ in range(self.h_layer)])
        self.gcn_layer = GCNLayer()

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.01)
        self.edgeDropper = SpAdjDropEdge()

        self.E_u, self.E_i = None, None
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.adj = adj

    def info_nce(self, embeds1, embeds2, nodes, temp, normal=True):
        view1 = embeds1[nodes]
        view2 = embeds2[nodes]
        if normal:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / temp
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def info_nce_ui(self, u_embeds, i_embeds, uid, pos, temp, normal=True):
        view1 = u_embeds[uid]
        view2 = i_embeds[pos]
        view3 = i_embeds
        if normal:
            view1, view2, view3 = F.normalize(view1, dim=1), F.normalize(view2, dim=1), F.normalize(view3, dim=1)
        pos_score = torch.sum(torch.exp((view1 @ view2.T) / temp), dim=1)
        all_score = torch.sum(torch.exp((view1 @ view3.T) / temp), dim=1)
        # print("shape:" + str((pos_score / all_score).shape) + "user:" + str(pos_score.shape))
        return -torch.log(pos_score / all_score).mean()

    def bpr_loss(self, u_embeds, i_embeds, uids, pos, neg):
        u_emb = u_embeds[uids]
        pos_emb = i_embeds[pos]
        neg_emb = i_embeds[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        p = ((pos_scores - neg_scores) / u_embeds.shape[-1]).sigmoid()
        # p = (pos_scores - neg_scores).sigmoid()
        loss_r = -(p + 1e-15).log().mean()
        return loss_r

    def forward(self, uids, iids, pos, neg, test=False):
        if test == True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1 - mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            embeds = torch.concat([self.u_embeds, self.i_embeds], dim=0)

            bn_lats, hbn_lats, bn_lats_2 = [], [], []
            lats, lats_1, lats_2, lats_3 = [embeds], [], [], []

            for i in range(self.layer):
                # embedding batch norm
                bn_lats.append(self.bn_layers[i](lats[-1]))

                # GCN
                user_embeds_1 = self.gcn_layer(self.adj_norm, bn_lats[-1][self.user:])
                item_embeds_1 = self.gcn_layer(self.adj_norm.transpose(0, 1), bn_lats[-1][:self.user])

                # user item connect
                embeds_1 = torch.cat([user_embeds_1, item_embeds_1], dim=0)

                # fusion
                lats_1.append(embeds_1)
                lats.append(embeds_1 + lats[-1])

            # lats.append(sum(lats))
            self.E_u = lats[-1][:self.user]
            self.E_i = lats[-1][self.user:]

            # bpr loss
            loss_r = self.bpr_loss(self.E_u, self.E_i, uids, pos, neg)

            loss_cl1 = 0
            for i in range(self.layer):
                embeds1 = lats[i + 1]
                embeds2 = lats[i]
                loss_cl1 += (self.info_nce(embeds1[:self.user], embeds2[:self.user], torch.unique(uids), self.temp_1) +
                             self.info_nce(embeds1[self.user:], embeds2[self.user:], torch.unique(pos), self.temp_1))

            loss_cl = (loss_cl1 * self.lambda_1) / self.layer

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.reg

            # total loss
            loss = loss_r + loss_cl + loss_reg
            return loss, loss_r, loss_cl, loss_reg


class ResGNN(nn.Module):
    def __init__(self, layer, user, item, dim):
        super(ResGNN, self).__init__()
        self.user, self.item = user, item
        self.layer = layer
        self.bn = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(self.layer)])
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(self.layer)])

    def forward(self, adj, embeds):
        lats, gcn_lats = [embeds], [embeds]
        bn_lats = []
        for i in range(self.layer):
            bn_lats.append(self.bn[i](lats[-1]))
            user_embeds = torch.spmm(adj, bn_lats[-1][self.user:])
            item_embeds = torch.spmm(adj.transpose(0, 1), bn_lats[-1][:self.user])
            embedding = torch.cat([user_embeds, item_embeds], dim=0)
            gcn_lats.append(embedding)
            lats.append(embedding + lats[-1])
        return lats, gcn_lats


class LightGCN(nn.Module):
    def __init__(self, layer, user, item, dim):
        super(LightGCN, self).__init__()
        self.user, self.item = user, item
        self.layer = layer

    def forward(self, adj, embeds):
        lats, gcn_lats = [embeds], [embeds]
        for i in range(self.layer):
            user_embeds = torch.spmm(adj, lats[-1][self.user:])
            item_embeds = torch.spmm(adj.transpose(0, 1), lats[-1][:self.user])
            embedding = torch.cat([user_embeds, item_embeds], dim=0)
            gcn_lats.append(embedding)
            lats.append(embedding + lats[-1])
        return lats, gcn_lats


class ResHGNN(nn.Module):
    def __init__(self, layer, user, item, dim):
        super(ResHGNN, self).__init__()
        self.user, self.item = user, item
        self.layer = layer
        self.bn = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(self.layer)])

    def forward(self, adj_user, adj_item, embeds):
        lats, gcn_lats = [embeds], [embeds]
        bn_lats = []
        for i in range(self.layer):
            bn_lats.append(self.bn[i](lats[-1]))
            user_hyper_edge_embeds = torch.mm(adj_user.T, bn_lats[-1][:self.user])
            user_embeds = torch.mm(adj_user, user_hyper_edge_embeds)
            item_hyper_edge_embeds = torch.mm(adj_item.T, bn_lats[-1][self.user:])
            item_embeds = torch.spmm(adj_item, item_hyper_edge_embeds)
            embedding = torch.cat([user_embeds, item_embeds], dim=0)
            gcn_lats.append(embedding)
            lats.append(embedding + lats[-1])
        return lats, gcn_lats


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


class HGNNLayer(nn.Module):
    def  __init__(self):
        super(HGNNLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj, embeddings):
        tmp = torch.mm(adj.T, embeddings)
        lat = torch.mm(adj, tmp)
        return lat


class HyperEncoder(nn.Module):
    def __init__(self):
        super(HyperEncoder, self).__init__()

    def forward(self, adj, embeddings):
        embedding, hyper_embedding = embeddings
        new_embedding = torch.mm(adj, hyper_embedding)
        new_hyper_embedding = torch.mm(adj.T, embedding)
        return new_embedding, new_hyper_embedding


class FNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, embeddings):
        return self.linear2(self.linear1(embeddings))


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
