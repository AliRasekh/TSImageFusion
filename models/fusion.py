import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
from .loss import RankingLoss, CosineLoss, KLDivLoss
import torch.nn.functional as F


def get_torch_trans(heads=4, layers=1, d_model=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=heads, dim_feedforward=1024, batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
	
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model


        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        

        if self.args.labels_set == 'radiology':
            target_classes = self.args.vision_num_classes
            lstm_in = self.cxr_model.feats_dim
            projection_in = self.ehr_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        feats_dim = 2 * self.ehr_model.feats_dim

        self.fused_cls = nn.Sequential(
            nn.Linear(feats_dim, self.args.num_classes),
            nn.Sigmoid()
        )

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()


        
        #self.attn_layer = nn.MultiheadAttention(lstm_in, num_heads=4, batch_first=True)
        self.attn_layer = get_torch_trans(heads=8, layers=2, d_model=lstm_in)

        self.attn_fused_cls =  nn.Sequential(
            nn.Linear(feats_dim, target_classes),
            nn.Sigmoid()
        )

        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        ) 

        if self.args.loss == 'focal_loss':
            self.attn_fused_cls =  nn.Sequential(
                nn.Linear(feats_dim, target_classes))
            self.lstm_fused_cls =  nn.Sequential(
                nn.Linear(lstm_out, target_classes))

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)

            
    def forward_uni_cxr(self, x, seq_lengths=None, img=None ):
        cxr_preds, _ , feats = self.cxr_model(img)
        return {
            'uni_cxr': cxr_preds,
            'cxr_feats': feats
            }
    
    def forward(self, x, seq_lengths=None, img=None, pairs=None ):
        if self.args.fusion_type == 'uni_cxr':
            return self.forward_uni_cxr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type in ['joint',  'early', 'late_avg', 'unified']:
            return self.forward_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )
        elif self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type == 'lstm':
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )

        elif self.args.fusion_type == 'uni_ehr_lstm':
            return self.forward_lstm_ehr(x, seq_lengths=seq_lengths, img=img, pairs=pairs )

        elif self.args.fusion_type == 'attention':
            return self.forward_attention_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs )
        
    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , feats = self.ehr_model(x, seq_lengths)
        return {
            'uni_ehr': ehr_preds,
            'ehr_feats': feats
            }

    def forward_fused(self, x, seq_lengths=None, img=None, pairs=None ):

        ehr_preds , ehr_feats = self.ehr_model(x, seq_lengths)
        cxr_preds, _ , cxr_feats = self.cxr_model(img)
        projected = self.projection(cxr_feats)

        # loss = self.align_loss(projected, ehr_feats)

        feats = torch.cat([ehr_feats, projected], dim=1)
        fused_preds = self.fused_cls(feats)

        # late_avg = (cxr_preds + ehr_preds)/2
        return {
            'early': fused_preds, 
            'joint': fused_preds, 
            'ehr_feats': ehr_feats,
            'cxr_feats': projected,
            'unified': fused_preds
            }

    def forward_attention_fused(self, x, seq_lengths=None, img=None, pairs=None ):
        if self.args.labels_set == 'radiology':
            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
            _, _ , cxr_feats = self.cxr_model(img)

            feats = cxr_feats[:,None,:]

            ehr_feats = self.projection(ehr_feats)

            ehr_feats[list(~np.array(pairs))] = 0
            feats = torch.cat([feats, ehr_feats[:,None,:]], dim=1)
        else:
            _ , ehr_feats = self.ehr_model(x, seq_lengths)            
            _, _ , cxr_feats = self.cxr_model(img)
            cxr_feats = self.projection(cxr_feats)

            cxr_feats[list(~np.array(pairs))] = 0
            if len(ehr_feats.shape) == 1:
                # print(ehr_feats.shape, cxr_feats.shape)
                # import pdb; pdb.set_trace()
                feats = ehr_feats[None,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
            else:
                feats = ehr_feats[:,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)

        
        feats = self.attn_layer(feats)
        #feats = attn_feats + feats
        
        feats = feats.reshape(feats.shape[0], -1)    
        fused_preds = self.attn_fused_cls(feats)

        return {
            'attention': fused_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': cxr_feats,
        }
    
        
        
    def forward_lstm_fused(self, x, seq_lengths=None, img=None, pairs=None ):
        if self.args.labels_set == 'radiology':
            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
            _, _ , cxr_feats = self.cxr_model(img)

            feats = cxr_feats[:,None,:]

            ehr_feats = self.projection(ehr_feats)

            ehr_feats[list(~np.array(pairs))] = 0
            feats = torch.cat([feats, ehr_feats[:,None,:]], dim=1)
        else:

            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
            _, _ , cxr_feats = self.cxr_model(img)
            cxr_feats = self.projection(cxr_feats)

            cxr_feats[list(~np.array(pairs))] = 0
            if len(ehr_feats.shape) == 1:
                feats = ehr_feats[None,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
            else:
                feats = ehr_feats[:,None,:]
                feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        seq_lengths = np.array([1] * len(seq_lengths))
        seq_lengths[pairs] = 2
                
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()
        
        fused_preds = self.lstm_fused_cls(out)
        

        return {
            'lstm': fused_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': cxr_feats,
        }
    
    def forward_lstm_ehr(self, x, seq_lengths=None, img=None, pairs=None ):
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        feats = ehr_feats[:,None,:]
        
        
        seq_lengths = np.array([1] * len(seq_lengths))
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()
        
        fused_preds = self.lstm_fused_cls(out)

        return {
            'uni_ehr_lstm': fused_preds,
        }
