import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from torch.cuda.amp import autocast as autocast, GradScaler
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            print('{:.1f}M'.format(tot / 1e6))
        else:
            print('{:.1f}K'.format(tot / 1e3))
    else:
        return None


class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        print('---------------model prepared---------------')
        compute_n_params(self.model)
        compute_n_params(self.para_model)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()

        return label, label_aux

    def make_nk_label(self, n, k, ep_per_batch=1):
        label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
        label = label.repeat(ep_per_batch)
        return label

    def train(self):
        pass

    def evaluate(self, data_loader):
        pass

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((10000, 2))  # loss and acc
        # label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        # label = label.type(torch.LongTensor)
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        log = open(self.args.log_path, 'a+')
        log.write('best epoch {}, best val acc={:.4f} + {:.4f} \n'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        log.close()
        with torch.no_grad():
            test_gen = tqdm(self.test_loader)
            tl1 = Averager()
            ta = Averager()
            for i, batch in enumerate(test_gen, 1):
                # for i, batch in enumerate(self.test_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                with autocast():
                    logits = self.model(data)
                    loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label) * 100
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
                tl1.add(loss)
                ta.add(acc)
                test_gen.set_description('测试阶段:CE_loss={:.4f} 平均acc={:.4f}'.format(tl1.item(), ta.item()))
        # print('测试阶段:平均loss1={:.4f} 平均acc={:.4f}'.format(tl1.item(), ta.item()))
        log = open(self.args.log_path, 'a+')
        log.write('测试阶段:CE_loss={:.4f} 平均acc={:.4f} \n'.format(tl1.item(), ta.item()))
        log.close()
        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval']))

        return vl, va, vap

    def final_record(self):
        # save the best performance in a txt file

        with open(
                osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])),
                'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))
