import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerInterEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            start_decay_steps=args.decay_step, decay_steps=5, lr_decay=0.99)
    # print([n for n, p in list(model.named_parameters())])
    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if optim.method == 'adam' and len(optim.optimizer.state) < 1:
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Bert(nn.Module):
    def __init__(self, temp_dir):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

    def forward(self, x, mask, segs):
        top_vec, _ = self.model(x, mask, segs)
        return top_vec


class Summarizer(nn.Module):
    def __init__(self, args, device):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir)
        self.regressor = nn.Linear(self.bert.model.config.hidden_size, 1, bias=True)
        self.weight_sigmoid = nn.Sigmoid()

        self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                               args.dropout, args.inter_layers)
        # self.sent_encoder = TransformerSenEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
        #                                        args.dropout, args.inter_layers)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, mask, segs)
        # print(top_vec.shape)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        cluster_vec = torch.mean(sents_vec, dim=1)
        x = self.regressor(cluster_vec)
        cluster_weight = self.weight_sigmoid(x).squeeze(-1)
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sents_vec, sent_scores, mask_cls, cluster_weight

nn.TransformerEncoder