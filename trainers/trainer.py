from __future__ import absolute_import
from __future__ import print_function

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime, timedelta
import time
from models.fusion import Fusion
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from .trainer import Trainer
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from .utils import get_model_performance
from models.loss import FocalLoss, MultiLoss, FocalLoss_learnable
from PIL import Image
from torchvision import transforms


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        # self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # Filter out all-zero samples from z_i and z_j
        all_zero_mask = torch.all(z_i == 0, dim=1) & torch.all(z_j == 0, dim=1)

        # Invert the mask to keep rows where at least one of z_i or z_j is non-zero
        keep_mask = ~all_zero_mask

        # Only keep non-zero samples in z_i and z_j
        z_i_non_zero = z_i[keep_mask]
        z_j_non_zero = z_j[keep_mask]

        N = 2 * len(z_i_non_zero) * self.world_size

        # Concatenate non-zero samples
        z = torch.cat((z_i_non_zero, z_j_non_zero), dim=0)

        # z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, len(z_i_non_zero) * self.world_size)
        sim_j_i = torch.diag(sim, -1 * len(z_i_non_zero) * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        self.mask = self.mask_correlated_samples(len(z_i_non_zero), self.world_size)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class Pre_Train_ehr(nn.Module):
    def __init__(self, ehr_model):
        super(Pre_Train_ehr, self).__init__()

        self.ehr_model = ehr_model

        self.inter_modality_header = nn.Sequential(
            nn.Linear(self.ehr_model.feats_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.intra_modality_header = nn.Sequential(
            nn.Linear(self.ehr_model.feats_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, ehr, ehr_aug, seq_lengths):
        scores, ehr_i = self.ehr_model(ehr, seq_lengths)
        scores, ehr_i_aug = self.ehr_model(ehr_aug, seq_lengths)

        inter_y = self.inter_modality_header(ehr_i)
        inter_y_aug = self.inter_modality_header(ehr_i_aug)
        intra_y = self.intra_modality_header(ehr_i)

        return inter_y, inter_y_aug, intra_y


class Pre_Train_cxr(nn.Module):
    def __init__(self, cxr_model):
        super(Pre_Train_cxr, self).__init__()

        self.cxr_model = cxr_model

        self.inter_modality_header = nn.Sequential(
            nn.Linear(self.cxr_model.feats_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.intra_modality_header = nn.Sequential(
            nn.Linear(self.cxr_model.feats_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, cxr, cxr_aug):
        preds, lossvalue_bce, cxr_i = self.cxr_model(cxr)
        preds, lossvalue_bce, cxr_i_aug = self.cxr_model(cxr_aug)

        inter_y = self.inter_modality_header(cxr_i)
        inter_y_aug = self.inter_modality_header(cxr_i_aug)
        intra_y = self.intra_modality_header(cxr_i)

        return inter_y, inter_y_aug, intra_y


class Trainer():
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl=None
        ):

        self.args = args
        self.time_start = time.time()
        self.time_end = time.time()
        self.start_epoch = 1
        self.patience = 0
        self.levels = np.array(['acute', 'acute' ,'acute' ,'mixed' ,'chronic' ,'chronic', 'acute', 'mixed', 'mixed' ,'chronic', 'mixed' ,'chronic', 
        'chronic' ,'chronic' ,'acute', 'acute', 'chronic' ,'mixed', 'acute' ,
        'acute', 'acute' ,'acute' ,'acute', 'acute' ,'acute'])
        
        
        self.epoch = 0 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.ehr_model = LSTM(input_dim=76, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)

        self.model = Fusion(args, self.ehr_model, self.cxr_model ).to(self.device)
        self.init_fusion_method()
        
        ######################################################################
        if self.args.contrastive == 'NT-Xent':
            self.ehr_model, self.cxr_model = self.Pre_Train(self.train_dl.batch_size, self.train_dl, self.val_dl, self.ehr_model, self.cxr_model, self.device)
        #######################################################################
        
        
        if self.args.loss == 'focal_loss':
            self.loss = FocalLoss(gamma=self.calc_gammas(self.val_dl),alpha=-1)
            self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
            print('loss-focal')
        elif self.args.loss == 'uncertainty_loss':
            self.loss = MultiLoss(num_tasks=self.args.num_classes, device=self.device)
            self.optimizer = optim.Adam(
                [
                    {"params" : self.model.parameters(), "betas" :(0.9, self.args.beta_1)},
                    {"params" :self.loss.parameters(), "lr" : 0.0007}
                ],
                args.lr
            )

            print('loss-uncertainty')
        elif self.args.loss == 'focal_loss_learnable':
            self.loss = FocalLoss_learnable(num_tasks=self.args.num_classes, device=self.device)
            self.optimizer = optim.Adam(
                [
                    {"params" : self.model.parameters(), "betas" :(0.9, self.args.beta_1)},
                    {"params" :self.loss.parameters(), "lr" : 0.0025}
                ],
                args.lr
            )

            print('loss-focal_learnable')
        else:
            self.loss = nn.BCELoss() 
            self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
            print('loss-bce')

        self.load_state()
        print(self.ehr_model)
        print(self.optimizer)
        print(self.loss)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99) 
        self.epochs_stats = {'loss train': [], 'loss val': [], 'auroc val': [], 'loss align train': [], 'loss align val': []}
        
        
        
    def calc_gammas(self, dl, gamma=2):
        print('num classes :',len(dl.dataset.CLASSES),' - num samples:',len(dl.dataset))
        support_vec = torch.zeros(len(dl.dataset.CLASSES)).to(self.device)
        for i, (ehr_data , cxr_data , ehr_labels, cxr_labels, seq_lengths, pairs) in enumerate (dl):
            ehr_labels = torch.from_numpy(ehr_labels).float().to(self.device)
            support_vec = support_vec + ehr_labels.sum(axis=0)
        treshold = 0.1
        print('support vec', support_vec)
        gammas = ((support_vec / len(dl.dataset)) < treshold) * gamma
        print('gammas are', gammas)
        return gammas
    
    def init_fusion_method(self):

        '''
        for early fusion
        load pretrained encoders and 
        freeze both encoders
        ''' 

        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        
        if self.args.load_state is not None:
            self.load_state()


        if 'uni_ehr' in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
        elif 'uni_cxr' in self.args.fusion_type:
            self.freeze(self.model.ehr_model)
        elif 'late' in self.args.fusion_type:
            self.freeze(self.model)
        elif 'early' in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
            self.freeze(self.model.ehr_model)
        elif 'lstm' in self.args.fusion_type or 'attention' in self.args.fusion_type:
            # self.freeze(self.model.cxr_model)
            # self.freeze(self.model.ehr_model)
            pass

    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)

        print('number of batches :', len(self.train_dl), 'number of data points :', len(self.train_dl.sampler))

        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs)
            
            pred = output[self.args.fusion_type].squeeze()
            loss = self.loss(pred, y)
            epoch_loss += loss.item()
            if self.args.align > 0.0:
                loss = loss + self.args.align * output['align_loss']
                epoch_loss_align = epoch_loss_align + self.args.align * output['align_loss'].item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:   
                if self.args.loss == 'uncertainty_loss':
                    print('grad is :',self.loss.log_var.grad)
                    print("loss_log_var: ", self.loss.log_var)
                    print("1/var^2 is: ", torch.exp(-self.loss.log_var))
                if self.args.loss == 'focal_loss_learnable':
                    print('log gamma is :',self.loss.log_gamma)
                    print('gamma is: ',torch.exp(self.loss.log_gamma))
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
        
        if self.args.loss == 'focal_loss' or self.args.loss == 'focal_loss_learnable':
            outPRED = F.sigmoid(outPRED) ##only when model doesn't have sigmoid layer
        
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        self.epochs_stats['loss train'].append(epoch_loss/i)
        self.epochs_stats['loss align train'].append(epoch_loss_align/i)
        return ret
    
    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                output = self.model(x, seq_lengths, img, pairs)
                
                pred = output[self.args.fusion_type]
                
                if self.args.fusion_type != 'uni_cxr' or self.args.task == 'in-hospital-mortality':
                    if len(pred.shape) > 1:
                         pred = pred.squeeze()
                
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                if self.args.align > 0.0:

                    epoch_loss_align +=  output['align_loss'].item()
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)

        
        self.scheduler.step(epoch_loss/len(self.val_dl))
        
        if self.args.loss == 'focal_loss' or self.args.loss == 'focal_loss_learnable':
            outPRED = F.sigmoid(outPRED) ##only when model doesn't have sigmoid layer
        
        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
        np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy()) 
        np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy()) 
        
        
        print(f'shape outPRED : {outPRED.shape} shape outGT: {outGT.shape}')
        print('number of batches :', len(dl), 'number of data points :', len(dl.sampler))
        
        self.computeF1(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy())   

        self.epochs_stats['auroc val'].append(ret['auroc_mean'])

        self.epochs_stats['loss val'].append(epoch_loss/i)
        self.epochs_stats['loss align val'].append(epoch_loss_align/i)

        return ret



    def compute_late_fusion(self, y_true, uniout_cxr, uniout_ehr):
        y_true = np.array(y_true)
        predictions_cxr = np.array(uniout_cxr)
        predictions_ehr = np.array(uniout_ehr)
        best_weights = np.ones(y_true.shape[-1])
        best_auroc = 0.0
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for class_idx in range(y_true.shape[-1]):
            for weight in weights:
                predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
                predictions[:, class_idx] = (predictions_ehr[:, class_idx] * weight) + (predictions_cxr[:, class_idx] * 1-weight)
                auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
                auroc_mean = np.mean(np.array(auc_scores))
                if auroc_mean > best_auroc:
                    best_auroc = auroc_mean
                    best_weights[class_idx] = weight
                # predictions = weight * predictions_cxr[]


        predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
        print(best_weights)

        auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
        ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                            average="micro")
        ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                            average="macro")
        ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                                average="weighted")
        
        best_stats = {"auc_scores": auc_scores,
                "ave_auc_micro": ave_auc_micro,
                "ave_auc_macro": ave_auc_macro,
                "ave_auc_weighted": ave_auc_weighted,
                "auroc_mean": np.mean(np.array(auc_scores))
                }
        self.print_and_write(best_stats , isbest=True, prefix='late fusion weighted average')

        return best_stats 

    def test(self):
        print('validating ... ')
        self.epoch = 0
        self.model.eval()
        ret = self.validate(self.val_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
        self.model.eval()
        ret = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        return

    def eval(self):
        print('validating ... ')
        self.epoch = 0
        self.model.eval()
        ret = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        return
    
    def train(self):
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            ret = self.validate(self.val_dl)
            self.save_checkpoint(prefix='last')

            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_checkpoint()
                # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
                self.print_and_write(ret, isbest=True)
                self.patience = 0
            else:
                self.print_and_write(ret, isbest=False)
                self.patience+=1

            self.model.train()
            self.train_epoch()
            self.plot_stats(key='loss', filename='loss.pdf')
            self.plot_stats(key='auroc', filename='auroc.pdf')
            if self.patience >= self.args.patience:
                break
        self.print_and_write(self.best_stats , isbest=True)

        
    def load_ehr_pheno(self, load_state):
        
        checkpoint = torch.load(load_state)
        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state or 'ehr_model' not in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

        print(f'loaded ehr checkpoint from {load_state}')

    def load_state(self):
        if self.args.load_state is None:
            return
        checkpoint = torch.load(self.args.load_state)


        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        print(f'loaded model checkpoint from {self.args.load_state}')

    def load_cxr_pheno(self, load_state):
        checkpoint = torch.load(load_state)

        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state or 'cxr_model' not in name:
                # print(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

        print(f'loaded cxr checkpoint from {load_state}')


    def freeze(self, model):
        for p in model.parameters():
           p.requires_grad = False
    def plot_array(self, array, disc='loss'):
        plt.plot(array)
        plt.ylabel(disc)
        plt.savefig(f'{disc}.pdf')
        plt.close()
       
    def computeF1(self, y_true, preds):
        if self.args.labels_set != 'radiology' and self.args.labels_set != 'pheno':
            y_pred_metric = np.round(preds)
            F1 = metrics.f1_score(y_true, y_pred_metric, average='macro')
            print(f'Accuracy : {metrics.accuracy_score(y_true, y_pred_metric)}, F1 macro Average : {F1}')
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_metric).ravel()    
            print(f'True Positive: {tp} , False Positive: {fp}, True Negative: {tn}, False Negative: {fn}')
            report = metrics.classification_report(y_true, y_pred_metric, digits=3)
            print(report)
        else:
            y_pred_metric = np.round(preds)
            F1_macro = metrics.f1_score(y_true, y_pred_metric, average='macro')
            F1_scores = metrics.f1_score(y_true, y_pred_metric, average=None)
            report = metrics.classification_report(y_true, y_pred_metric, digits=3)
            print(report)
            avg_f1 = 0
            for i in range(y_true.shape[1]):
                df = pd.DataFrame({'y_truth': y_true[:, i], 'y_pred': y_pred_metric[:, i]})
                F1 = metrics.f1_score(df['y_truth'], df['y_pred'], average='macro')
                accuracy = metrics.accuracy_score(df['y_truth'], df['y_pred'])
                avg_f1 += F1
                print(f'{i} - {self.val_dl.dataset.CLASSES[i]} - Accuracy: {accuracy}, F1 macro Average : {F1}')
            avg_f1 /= y_true.shape[1]
            print('averge macro f1 is ', avg_f1)
    
    def computeAUROC(self, y_true, predictions, verbose=1):        
        y_true = np.array(y_true)
        predictions = np.array(predictions)
        auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
        ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                            average="micro")
        ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                            average="macro")
        ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                                average="weighted")

        auprc = metrics.average_precision_score(y_true, predictions, average=None)

        
        auc_scores = []
        auprc_scores = []
        ci_auroc = []
        ci_auprc = []
        if len(y_true.shape) == 1:
            y_true = y_true[:, None]
            predictions = predictions[:, None]
        for i in range(y_true.shape[1]):
            df = pd.DataFrame({'y_truth': y_true[:, i], 'y_pred': predictions[:, i]})
            (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc) = get_model_performance(df)
            auc_scores.append(test_auroc)
            auprc_scores.append(test_auprc)
            ci_auroc.append((lower_auroc, upper_auroc))
            ci_auprc.append((lower_auprc, upper_auprc))
        
        auc_scores = np.array(auc_scores)
        auprc_scores = np.array(auprc_scores)
       
        return { "auc_scores": auc_scores,
            
            "auroc_mean": np.mean(auc_scores),
            "auprc_mean": np.mean(auprc_scores),
            "auprc_scores": auprc_scores, 
            'ci_auroc': ci_auroc,
            'ci_auprc': ci_auprc,
            }


    def step_lr(self, epoch):
        step = self.steps[0]
        for index, s in enumerate(self.steps):
            if epoch < s:
                break
            else:
                step = s

        lr = self.args.lr * (0.1 ** (epoch // step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_eta(self, epoch, iter):
        # import pdb; pdb.set_trace()
        done_epoch = epoch - self.start_epoch
        remaining_epochs = self.args.epochs - epoch

        iter +=1
        self.time_end = time.time()
        
        delta = self.time_end - self.time_start
        
        done_iters = len(self.train_dl) * done_epoch + iter
        
        remaining_iters = len(self.train_dl) * remaining_epochs - iter

        delta = (delta/done_iters)*remaining_iters
        
        sec = timedelta(seconds=int(delta))
        d = (datetime(1,1,1) + sec)
        eta = f"{d.day-1} Days {d.hour}:{d.minute}:{d.second}"

        return eta
    def get_gt(self, y_ehr, y_cxr):
        if 'radiology' in self.args.labels_set :
            return y_cxr
        else:
            return torch.from_numpy(y_ehr).float()

    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            'best_auroc': self.best_auroc, 
            'optimizer' : self.optimizer.state_dict(),
            'epochs_stats': self.epochs_stats
            }, path)
        print(f"saving {prefix} checkpoint at epoch {self.epoch}")

    def plot_stats(self, key='loss', filename='training_stats.pdf'):
        for loss in self.epochs_stats:
            if key in loss:
                plt.plot(self.epochs_stats[loss], label = f"{loss}")
        
        plt.xlabel('epochs')
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.savefig(f"{self.args.save_dir}/{filename}")
        plt.close()
    def print_and_write(self, ret, prefix='val', isbest=False, filename='results.txt'):

        with open(f"{self.args.save_dir}/{filename}", 'a') as results_file:
            if isbest:
                
                ci_auroc_all = []
                ci_auprc_all = []

                if len(ret['auc_scores'].shape) > 0:
                    
                    for index, class_auc in enumerate(ret['auc_scores']):
                        # line = f'{self.val_dl.dataset.CLASSES[index]: <90} & {class_auc:0.3f} & {ret["auprc_scores"][index]:0.3f} ' 
                        line = f'{self.val_dl.dataset.CLASSES[index]: <90} & {class_auc:0.3f}({ret["ci_auroc"][index][1]:0.3f}, {ret["ci_auroc"][index][0]:0.3f}) & {ret["auprc_scores"][index]:0.3f} ({ret["ci_auprc"][index][1]:0.3f}, {ret["ci_auprc"][index][0]:0.3f}) ' 
                        ci_auroc_all.append([ret["ci_auroc"][index][0] , ret["ci_auroc"][index][1]])
                        ci_auprc_all.append([ret["ci_auprc"][index][0] , ret["ci_auprc"][index][1]])
                        print(line)
                        results_file.write(line)
                    
                else:

                    ci_auroc_all.append([ret["ci_auroc"][0][0] , ret["ci_auroc"][0][1]])
                    ci_auprc_all.append([ret["ci_auprc"][0][0] , ret["ci_auprc"][0][1]])

                ci_auroc_all = np.array(ci_auroc_all)
                ci_auprc_all = np.array(ci_auprc_all)

                auc_scores = ret['auc_scores']
                auprc_scores = ret['auprc_scores']

                accute_aurocs = np.mean(auc_scores) if self.args.labels_set != 'pheno' else np.mean(auc_scores[self.levels == 'acute'])
                mixed_aurocs = np.mean(auc_scores) if self.args.labels_set != 'pheno' else np.mean(auc_scores[self.levels == 'mixed'])
                chronic_aurocs = np.mean(auc_scores) if self.args.labels_set != 'pheno' else np.mean(auc_scores[self.levels == 'chronic'])
                
                accute_auprc = np.mean(auprc_scores) if self.args.labels_set != 'pheno' else np.mean(auprc_scores[self.levels == 'acute'])
                mixed_auprc = np.mean(auprc_scores) if self.args.labels_set != 'pheno' else np.mean(auprc_scores[self.levels == 'mixed'])
                chronic_auprc = np.mean(auprc_scores) if self.args.labels_set != 'pheno' else np.mean(auprc_scores[self.levels == 'chronic'])


                accute_aurocs_ci = np.mean(ci_auroc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auroc_all[self.levels == 'acute'], axis=0)
                mixed_aurocs_ci = np.mean(ci_auroc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auroc_all[self.levels == 'mixed'], axis=0)
                chronic_aurocs_ci = np.mean(ci_auroc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auroc_all[self.levels == 'chronic'], axis=0)
                
                accute_auprc_ci = np.mean(ci_auprc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auprc_all[self.levels == 'acute'], axis=0)
                mixed_auprc_ci = np.mean(ci_auprc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auprc_all[self.levels == 'mixed'], axis=0)
                chronic_auprc_ci = np.mean(ci_auprc_all, axis=0) if self.args.labels_set != 'pheno' else np.mean(ci_auprc_all[self.levels == 'chronic'], axis=0)


                line = f"\n\n\n{prefix}  {self.epoch:<3} best mean auc :{ret['auroc_mean']:0.3f} mean auprc {ret['auprc_mean']:0.3f} \n\n\n\
                    CI AUROC ({np.mean(ci_auroc_all[:, 0]):0.3f}, {np.mean(ci_auroc_all[:, 1]):0.3f}) CI AUPRC ({np.mean(ci_auprc_all[:, 0]):0.3f}, {np.mean(ci_auprc_all[:, 1]):0.3f}) \n\n\n \
                    AUROC accute {accute_aurocs:0.3f} mixed {mixed_aurocs:0.3f} chronic {chronic_aurocs:0.3f}\n\n\n \
                    AUROC accute CI ({accute_aurocs_ci[0]:0.3f}, {accute_aurocs_ci[1]:0.3f}) mixed ({mixed_aurocs_ci[0]:0.3f} , {mixed_aurocs_ci[1]:0.3f}) chronic ({chronic_aurocs_ci[0]:0.3f}, {chronic_aurocs_ci[1]:0.3f})\n\n\n \
                    AUPRC accute  {accute_auprc:0.3f} mixed {mixed_auprc:0.3f} chronic {chronic_auprc:0.3f} \n\n\n \
                    AUPRC accute CI  ({accute_auprc_ci[0]:0.3f}, {accute_auprc_ci[1]:0.3f}) mixed ({mixed_auprc_ci[0]:0.3f},  {mixed_auprc_ci[1]:0.3f}) chronic ({chronic_auprc_ci[0]:0.3f}, {chronic_auprc_ci[1]:0.3f}) \n\n\n\
                    " 
                print(line)
                results_file.write(line)
            else:
                line = f"\n\n\n{prefix}  {self.epoch:<3} mean auc :{ret['auroc_mean']:0.6f} mean auprc {ret['auprc_mean']:0.6f}\n\n\n " 
                print(line)
                results_file.write(line)

    def augmentation_transforms_1(self, x):

        x = transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0))(x)

        x = transforms.ColorJitter(brightness=0.08)(x)

        x = transforms.RandomHorizontalFlip()(x)

        x = transforms.RandomRotation(degrees=40)(x)

        x = transforms.RandomVerticalFlip()(x)

        return x

    def augmentation_transforms_2(self, x):

        x = transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0))(x)

        x = transforms.RandomHorizontalFlip()(x)

        x = transforms.RandomRotation(degrees=40)(x)

        x = transforms.RandomVerticalFlip()(x)

        x = transforms.RandomHorizontalFlip()(x)

        x = transforms.RandomRotation(degrees=40)(x)

        x = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(x)

        return x

    def Pre_Train(self, batch_size, train_dl, val_dl, LSTM, CXRModels, device):

        temperature = .5

        inter_cxr_criterion_contrastive = NT_Xent(batch_size, temperature, world_size=1)
        inter_ehr_criterion_contrastive = NT_Xent(batch_size, temperature, world_size=1)
        intra_criterion_contrastive = NT_Xent(batch_size, temperature, world_size=1)

        augmentation_transforms_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=40),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augmentation_transforms_2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=40),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # get to models to pretrain!
        model_ehr = Pre_Train_ehr(LSTM)
        model_cxr = Pre_Train_cxr(CXRModels)

        model_ehr = model_ehr.to(device)
        model_cxr = model_cxr.to(device)

        optimizer = torch.optim.Adam([{'params': model_ehr.parameters()}, {'params': model_cxr.parameters()}],
                                     lr=0.0001)

        inter_cxr_loss_train = []
        inter_cxr_loss_val = []

        inter_ehr_loss_train = []
        inter_ehr_loss_val = []

        intra_loss_train = []
        intra_loss_val = []

        best_loss = 1000

        hold_ehr = None
        hold_cxr = None

        for epoch in range(100):  # todo: set number of epochs!

            model_ehr.train()
            model_cxr.train()
            for i, (ehr_data, cxr_data, ehr_labels, cxr_labels, seq_lengths, pairs) in enumerate(train_dl):
                # Forward pass
                model_ehr.train()
                model_cxr.train()

                print(cxr_data.shape)

                # transfer data to gpu
                ehr_data = torch.from_numpy(ehr_data).float()
                ehr_data = ehr_data.to(self.device)
                cxr_data = cxr_data.to(device)  # , cxr_labels.to(device)

                # add augmentation
                noise = torch.randn_like(ehr_data) * 0.01
                ehr_data_aug = ehr_data + noise

                ehr_data = ehr_data.to(torch.float32)
                ehr_data_aug = ehr_data_aug.to(torch.float32)

                ehr_data_aug = ehr_data_aug.to(device)

                cxr_data_aug_1 = self.augmentation_transforms_1(cxr_data)
                cxr_data_aug_2 = self.augmentation_transforms_2(cxr_data)

                cxr_data_aug_1 = cxr_data_aug_1.to(device)
                cxr_data_aug_2 = cxr_data_aug_2.to(device)

                # get the model outputs
                inter_ehr_outputs, inter_ehr_outputs_aug, intra_ehr_outputs = model_ehr(ehr_data, ehr_data_aug,
                                                                                        seq_lengths)
                inter_cxr_outputs, inter_cxr_outputs_aug, intra_cxr_outputs = model_cxr(cxr_data_aug_1, cxr_data_aug_2)

                print(f'shape of inter_ehr_outputs:{inter_ehr_outputs.shape}')
                print(f'shape of inter_ehr_outputs_aug:{inter_ehr_outputs_aug.shape}')
                print(f'shape of intra_ehr_outputs:{intra_ehr_outputs.shape}')
                print(f'shape of inter_cxr_outputs:{inter_cxr_outputs.shape}')
                print(f'shape of inter_cxr_outputs_aug:{inter_cxr_outputs_aug.shape}')
                print(f'shape of intra_cxr_outputs:{intra_cxr_outputs.shape}')

                if (len(inter_ehr_outputs) > batch_size - 1):

                    for i in range(batch_size):
                        # Check if the entire sample in cxr_data is all zero
                        if torch.all(cxr_data[i] == 0):
                            print('hahahahahahahah')
                            # Set the corresponding row in inter_ehr_outputs to zero
                            inter_cxr_outputs[i] = 0
                            inter_cxr_outputs_aug[i] = 0

                            intra_ehr_outputs[i] = 0
                            intra_cxr_outputs[i] = 0

                    # calculate the loss
                    inter_ehr_contrastive = inter_ehr_criterion_contrastive(inter_ehr_outputs, inter_ehr_outputs_aug)
                    inter_cxr_contrastive = inter_cxr_criterion_contrastive(inter_cxr_outputs, inter_cxr_outputs_aug)
                    intra_contrastive = intra_criterion_contrastive(intra_ehr_outputs, intra_cxr_outputs)

                    print(f'shape of inter_ehr_contrastive:{inter_ehr_contrastive}')
                    print(f'shape of inter_cxr_contrastive:{inter_cxr_contrastive}')
                    print(f'shape of intra_contrastive:{intra_contrastive}')

                    intra_contrastive_ = intra_contrastive.clone()
                    sum_loss_ehr = .5 * inter_ehr_contrastive + .5 * intra_contrastive

                    if torch.all(cxr_data == 0):
                        sum_loss_cxr = 0
                    else:
                        print('done')
                        print(f'sum is:{torch.sum(cxr_data)}')
                        sum_loss_cxr = .5 * inter_cxr_contrastive + .5 * intra_contrastive_
                        print(f'shape of sum_loss_cxr:{sum_loss_cxr.shape}')

                    loss = .5 * sum_loss_cxr + .5 * sum_loss_ehr

                    # store the loss
                    inter_ehr_loss_train.append(inter_ehr_contrastive.item())
                    inter_cxr_loss_train.append(inter_cxr_contrastive.item())
                    intra_loss_train.append(intra_contrastive.item())

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model_ehr.eval()
            model_cxr.eval()

            with torch.no_grad():

                total_ = 0

                for i, (ehr_data, cxr_data, ehr_labels, cxr_labels, seq_lengths, pairs) in enumerate(val_dl):

                    # transfer data to gpu
                    ehr_data = torch.from_numpy(ehr_data).float()
                    ehr_data = ehr_data.to(self.device)
                    cxr_data = cxr_data.to(device)

                    # add augmentation
                    noise = torch.randn_like(ehr_data) * 0.1
                    ehr_data_aug = ehr_data + noise

                    ehr_data = ehr_data.to(torch.float32)
                    ehr_data_aug = ehr_data_aug.to(torch.float32)
                    ehr_data_aug = ehr_data_aug.to(device)

                    cxr_data_aug_1 = self.augmentation_transforms_1(cxr_data)
                    cxr_data_aug_2 = self.augmentation_transforms_2(cxr_data)

                    cxr_data_aug_1 = cxr_data_aug_1.to(device)
                    cxr_data_aug_2 = cxr_data_aug_2.to(device)

                    # get the model outputs
                    inter_ehr_outputs, inter_ehr_outputs_aug, intra_ehr_outputs = model_ehr(ehr_data, ehr_data_aug,
                                                                                            seq_lengths)
                    inter_cxr_outputs, inter_cxr_outputs_aug, intra_cxr_outputs = model_cxr(cxr_data_aug_1,
                                                                                            cxr_data_aug_2)

                    if (len(inter_ehr_outputs) > batch_size - 1):
                        # calculate the loss
                        inter_ehr_contrastive = inter_ehr_criterion_contrastive(inter_ehr_outputs,
                                                                                inter_ehr_outputs_aug)
                        inter_cxr_contrastive = inter_cxr_criterion_contrastive(inter_cxr_outputs,
                                                                                inter_cxr_outputs_aug)
                        intra_contrastive = intra_criterion_contrastive(intra_ehr_outputs, intra_cxr_outputs)

                        total_ = total_ + inter_ehr_contrastive + inter_cxr_contrastive + intra_contrastive

                        # store the loss
                        inter_ehr_loss_val.append(inter_ehr_contrastive.item())
                        inter_cxr_loss_val.append(inter_cxr_contrastive.item())
                        intra_loss_val.append(intra_contrastive.item())

            # print the epoch and the losses
            print(f'Epoch {epoch}:\n'
                  f'inter_ehr_loss_train: {np.mean(inter_ehr_loss_train)}\n'
                  f'inter_cxr_loss_train: {np.mean(inter_cxr_loss_train)}\n'
                  f'intra_loss_train: {np.mean(intra_loss_train)}\n'
                  f'inter_ehr_loss_val: {np.mean(inter_ehr_loss_val)}\n'
                  f'inter_cxr_loss_val: {np.mean(inter_cxr_loss_val)}\n'
                  f'intra_loss_val: {np.mean(intra_loss_val)}\n')

        return model_ehr.ehr_model, model_cxr.cxr_model
