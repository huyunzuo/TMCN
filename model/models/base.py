import torch
import torch.nn as nn
import numpy as np


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10,
                                       0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            return (torch.Tensor(np.arange(args.way * args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way * args.shot, args.way * (args.shot + args.query))).long().view(1,
                                                                                                                   args.query,
                                                                                                                   args.way))
        else:
            return (
            torch.Tensor(np.arange(args.eval_way * args.eval_shot)).long().view(1, args.eval_shot, args.eval_way),
            torch.Tensor(np.arange(args.eval_way * args.eval_shot,
                                   args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query,
                                                                                                    args.eval_way))

    def split_shot_query(self, data, ep_per_batch=1):
        args = self.args  # 加上旋转的torch.Size([80, 4, 3, 84, 84])
        img_shape = data.shape[1:]
        data = data.view(ep_per_batch, args.way, args.shot + args.query, *img_shape)
        x_shot, x_query = data.split([args.shot, args.query], dim=2)
        x_shot = x_shot.contiguous()
        x_query = x_query.contiguous().view(ep_per_batch, args.way * args.query, *img_shape)
        return x_shot, x_query

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x_shot, x_query = self.split_shot_query(x, self.args.batch)
            shot_shape = x_shot.shape[:-3]
            query_shape = x_query.shape[:-3]
            img_shape = x_shot.shape[-3:]

            x_shot = x_shot.view(-1, *img_shape)
            x_query = x_query.view(-1, *img_shape)
            x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))  ##torch.Size([320, 512])
            fea_shape = x_tot.shape[1:]
            x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
            x_shot = x_shot.view(*shot_shape, *fea_shape)
            x_query = x_query.view(*query_shape, *fea_shape)
            if self.training:
                logits, logits_reg = self._forward(x_shot, x_query)
                return logits, logits_reg
            else:
                logits = self._forward(x_shot, x_query)
                return logits

    def _forward(self, x_shot, x_query):
        raise NotImplementedError('Suppose to be implemented by subclass')
