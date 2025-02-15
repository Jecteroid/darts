# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging


def broadcast_list(l, device_ids):
    """Broadcasting list"""
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i + len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class CNN_Structure(nn.Module):
    """CNN model"""

    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C : # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # in first cell, stem is used for both s0 & s1
        # C_pp & C_p is output channel size // C_cur is input channel size

        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):

            # if layer setting is 2, then set one is normal and another is reduce.
            if n_layers == 2:
                if i == 1:
                    C_cur *= 2
                    reduction = True
                else:
                    reduction = False

            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            elif i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class SearchCNNController(nn.Module):
    """SearchCNN controller supporting multi-gpu"""

    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3, device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize arch parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = CNN_Structure(C_in, C, n_classes, n_layers, self.n_nodes, stem_multiplier)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce)  # 보통은 여기서 끄읕

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y, self.complexity())

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("########--Alpha--########")
        logger.info("## Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n## Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("#########################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def n_operations(self):
        gene_ops = 0
        gene_ops_1 = 0

        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)

        for i in gene_normal:
            for j in i:
                gene_ops += 1 if j[0] != 'skip_connect' else 0
                gene_ops_1 += 1 if j[0] != 'skip_connect' else 0

        print('Normal ops: ', gene_ops_1)

        gene_ops_1 = 0

        for i in gene_reduce:
            for j in i:
                gene_ops += 1 if j[0] != 'skip_connect' else 0
                gene_ops_1 += 1 if j[0] != 'skip_connect' else 0

        print('Reduce ops: ', gene_ops_1)

        return gene_ops

    def complexity(self):
        c_normal = torch.tensor(0., requires_grad=True)
        c_reduce = torch.tensor(0., requires_grad=True)

        source_operations = 7.0
        nth_operations_max = 8.0
        nth_operations_min = 6.0

        c_max = torch.tensor(1.0 / source_operations * nth_operations_max, requires_grad=True) / 2.0

        c_min = torch.tensor(1.0 / source_operations * nth_operations_min, requires_grad=True) / 2.0

        fines = torch.tensor([1., 1., 0., 1., 1., 1., 1.]).cuda()

        print("## Alpha - normal")

        for a in self.alpha_normal:
            a = a[:, :-1]

            indx = torch.topk(torch.max(a, dim=1, keepdims=True)[0], 2, dim=0)[1][:, 0]

            max_mask = torch.zeros_like(a).scatter(1, torch.argmax(a, dim=1, keepdims=True), 1)

            a = F.softmax(a, dim=-1)

            tmp_1 = a * max_mask

            print(tmp_1[indx])

            with_fines = torch.sum(tmp_1[indx], dim=0) * fines

            c_normal = c_normal + torch.sum(with_fines)

        print("## Alpha - reduce")

        for a in self.alpha_reduce:
            a = a[:, :-1]

            indx = torch.topk(torch.max(a, dim=1, keepdims=True)[0], 2, dim=0)[1][:, 0]

            max_mask = torch.zeros_like(a).scatter(1, torch.argmax(a, dim=1, keepdims=True), 1)

            a = F.softmax(a, dim=-1)

            tmp_1 = a * max_mask

            print(tmp_1[indx])

            with_fines = torch.sum(tmp_1[indx], dim=0) * fines

            c_reduce = c_reduce + torch.sum(with_fines)

        # c_normal = c_normal if c_normal - c_clip >= torch.tensor(0.1, requires_grad=True) \
        #     else torch.tensor(0.1, requires_grad=True)

        # c_reduce = c_reduce if c_reduce - c_clip >= torch.tensor(0.1, requires_grad=True) \
        #     else torch.tensor(0.1, requires_grad=True)

        c_normal = torch.tensor(0.1, requires_grad=True) \
            if c_min <= c_normal <= torch.tensor(c_max + 0.1, requires_grad=True) \
            else c_normal - c_max if c_normal > torch.tensor(c_max + 0.1, requires_grad=True) \
            else c_min - c_normal

        c_reduce = torch.tensor(0.1, requires_grad=True) \
            if c_min <= c_reduce <= torch.tensor(c_max + 0.1, requires_grad=True) \
            else c_reduce - c_max if c_reduce > torch.tensor(c_max + 0.1, requires_grad=True) \
            else c_min - c_reduce

        print('complexity normal: ', c_normal)
        print('complexity reduce: ', c_reduce)

        print('complexity: ', c_normal + c_reduce)
        self.n_operations()

        return c_normal + c_reduce

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        print('\ngene_normal\n', gene_normal, '\ngene_normal\n')

        gene_normal_ops = 0

        for i in gene_normal:
            for j in i:
                gene_normal_ops += 1 if j[0] != 'skip_connect' else 0

        print(f'\operations : {gene_normal_ops} \n')

        gene_reduce_ops = 0

        print('\ngene_reduce\n', gene_reduce, '\ngene_reduce\n')

        for i in gene_reduce:
            for j in i:
                gene_reduce_ops += 1 if j[0] != 'skip_connect' else 0

        print(f'\operations : {gene_reduce_ops} \n')

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p






