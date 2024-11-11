import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--layer_after', default=4, type=int, help='apply mmtm module after fourth layer -1 indicates mmtm after every layer')
    parser.add_argument('--layers', default=1, type=int, help='number of lstm stacked modules')
    parser.add_argument('--vision_num_classes', default=14, type=int, help='number of classes')

    parser.add_argument('--resize', default=256, type=int, help='number of epochs to train')
    parser.add_argument('--crop', default=224, type=int, help='number of epochs to train')

    parser.add_argument('--vision-backbone', default='densenet121', type=str, help='[vit, resnet34, densenet121, densenet169, densenet201]')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',  help='load imagenet pretrained model')
    parser.add_argument('--eval', dest='eval', action='store_true',  help='eval the pretrained models on val and test split')
    parser.add_argument('--contrastive', default='no', type=str, help='[no, NT-Xent]')

    parser.add_argument('--network', type=str)
    parser.add_argument('--fusion_type', type=str, default='fused_ehr', help='train or eval for [attention, fused_ehr, fused_cxr, uni_cxr, uni_ehr]')
    parser.add_argument('--task', type=str, default='phenotyping', help='train or eval for in-hospital-mortality or phenotyping, decompensation, length-of-stay')
    parser.add_argument('--labels_set', type=str, default='pheno', help='pheno, radiology')

    parser.add_argument('--fusion', type=str, default='joint', help='train or eval for [early late joint]')

    parser.add_argument('--loss', type=str, default='BCE', help='BCE or focal_loss or focal_loss_learnable or uncertainty_loss')

    parser.add_argument('--dim', type=int, default=256,
                        help='number of hidden units')
    parser.add_argument('--depth', type=int, default=1,
                        help='number of bi-LSTMs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
    parser.add_argument('--load_state', type=str, default=None, help='state dir path')
    parser.add_argument('--load_state_cxr', type=str, default=None, help='state dir path')
    parser.add_argument('--load_state_ehr', type=str, default=None, help='state dir path')
    parser.add_argument('--mode', type=str, default="train",
                        help='mode: train or test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resume', dest='resume', help='resume training from state to load', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument('--patience', type=int, default=45, help='number of epoch to wait for best')

    parser.add_argument('--rec_dropout', type=float, default=0.0,
                        help="dropout rate for recurrent connections")
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--align', type=float, default=0.0, help='align weight')

    parser.add_argument('--data_pairs', type=str, default='paired', help='paired, ehr, cxr')
    parser.add_argument('--missing_token', type=str, default=None, help='zeros, learnable')

    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta_1 param for Adam optimizer')
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                                'use one of the provided ones.')

    parser.add_argument('--ehr_data_dir', type=str, help='Path to the data of phenotyping fusion_type',
                        default='../Data/mimic-iv-extracted/data')

#    parser.add_argument('--ehr_data_dir', type=str, help='Path to the data of phenotyping fusion_type',
#                        default='../Data_noisy/mimic-iv-extracted/data_0.1_1')

    parser.add_argument('--cxr_data_dir', type=str, help='Path to the data of phenotyping fusion_type',
                        default='../Data/mimic-cxr-jpg-2.0.0.physionet.org')

#    parser.add_argument('--cxr_data_dir', type=str, help='Path to the data of phenotyping fusion_type',
#                        default='../Data_noisy/mimic-cxr-jpg-2.0.0.physionet.org_CLAHE/mimic-cxr-jpg-2.0.0.physionet.org')

    parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='checkpoints')


    return parser
