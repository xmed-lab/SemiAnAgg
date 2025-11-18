import argparse
import os
import time
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='med_classify_dataset/skin',
                        help='dataset root dir')




    parser.add_argument('--cls_batch_size', type=int, default=64, help='batch_size per gpu')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu')
    parser.add_argument('--remove_n', type=int, default=0, help='remove_n')

    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
    parser.add_argument('--base_lr', type=float, default=2e-4,
                        help='maximum epoch number to train')  # adam:2e-4 sgd:2e-3 adamw:2e-3?
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--local_ep', type=int, default=1, help='local epoch')
    parser.add_argument('--num_users', type=int, default=10, help='local epoch')
    parser.add_argument('--num_labeled', type=int, default=1, help='local epoch')
    parser.add_argument('--rounds', type=int, default=200, help='local epoch')

    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--ssl_model', type=str, default='barlow', help='The log file name')

    parser.add_argument('--logdir', type=str, default='logs/', help='The log file name')
    parser.add_argument('--opt', type=str, default='sgd', help='sgd or adam or adamw')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'skin', 'SVHN', 'cifar100'], default='SVHN',
                        help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--model', type=str, default='Res18', help='neural network used in training')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--warmup_comm_round', type=int, default=256, help='the output dimension for the projection layer')

    ### tune
    parser.add_argument('--resume', '-r',default=True, action='store_true', help='resume from checkpoint')
    parser.add_argument('--Pretrained', '-Pretrained', action='store_true', help='resume from checkpoint')


    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    #parser.add_argument('--weight_decay', dest="weight_decay", default=0.02, type=float, help='weight decay')

    ### costs
    parser.add_argument('--label_uncertainty', type=str, default='U-Ones', help='label type')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=30, help='consistency_rampup')
    parser.add_argument('--lambda_u', type=float, default=0.02, help='start_epoch')

    ### unlabeled client training parameters
    parser.add_argument('--num-warmup-epochs',
                        '--num-warm-up-epochs',
                        dest="num_warmup_epochs",
                        default=0,
                        type=int,
                        help='number of warm-up epochs for unsupervised loss ramp-up during training'
                             'set to 0 to disable ramp-up')

    parser.add_argument('--lr-step-size',
                        '--learning-rate-step-size',
                        dest="lr_step_size",
                        default=5,
                        type=int,
                        help='step size for step learning rate decay')
    parser.add_argument("--max_grad_norm",
                        dest="max_grad_norm",
                        type=float,
                        default=5,
                        help="max gradient norm allowed (used for gradient clipping)")
    ### unsupervised loss
    parser.add_argument('--conf-threshold',
                        '--confidence-threshold',
                        dest="confidence_threshold",
                        default=0.9,
                        type=float,
                        help="confidence threshold for pair loss and unsupervised loss")
    ###
    parser.add_argument('--test', action='store_true', help='resume from checkpoint')
    parser.add_argument('--warmup', action='store_true', help='resume from checkpoint')
    parser.add_argument('--pl', action='store_true', help='resume from checkpoint')

    parser.add_argument('--vis_collapse', action='store_true', help='resume from checkpoint')

    parser.add_argument('--vis_ph', action='store_true', help='resume from checkpoint')
    parser.add_argument('--timm_cos', action='store_true', help='resume from checkpoint')
    parser.add_argument('--thresh_warmup', action='store_true',default=True, help='resume from checkpoint')

    parser.add_argument('--flex', action='store_true',default=True,help='resume from checkpoint')
    parser.add_argument('--fix', action='store_true', help='resume from checkpoint')
    parser.add_argument('--fed_flex', action='store_true', help='resume from checkpoint')

    parser.add_argument('--tau1', default=2, type=float,
                        help='tau for head1 consistency')
    parser.add_argument('--tau12', default=2, type=float,
                        help='tau for head2 consistency')
    parser.add_argument('--tau2', default=2, type=float,
                        help='tau for head2 balanced CE loss')
    parser.add_argument('--ema_u', default=0.9, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')
    parser.add_argument('--T', default=1.0, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')

    ### meta
    parser.add_argument('--meta_round', type=int, default=3, help='number of sub-consensus groups')
    parser.add_argument('--est_epoch', type=int, default=0, help='number of sub-consensus groups')


    parser.add_argument('--long_tailed', action='store_true', help='resume from checkpoint')

    parser.add_argument('--random_ssl', default=True,action='store_true',
                        help='whether the warm-up checkpoint is trained only on labeled client')

    parser.add_argument('--ssl_pretrain', action='store_true',
                        help='whether the warm-up checkpoint is trained only on labeled client')

    parser.add_argument('--unsup_lr', type=float, default=0.02,
                        help='lr of unsupervised clients')
    parser.add_argument('--loss_fn_name', type=str, default='BSM', metavar='N',
                        help='Options are: CE, BSM, focal, CB')
    parser.add_argument('--load_path', type=str, default='CE', metavar='N',
                        help='Options are: CE, BSM, focal, CB')

    parser.add_argument('--lambd', type=float, default=0.0051, help='sc')
    parser.add_argument('--scale_loss', type=float, default=0, help='sc')

    parser.add_argument('--main_T', type=float, default=0.9, help='sc')
    parser.add_argument('--ema_value', type=float, default=0.996, help='sc')
    parser.add_argument('--num_global_clusters', type=int, default=128, help='actual input size')
    parser.add_argument('--num_local_clusters', type=int, default=16, help='actual input size')
    parser.add_argument('--cluster_m_size', type=int, default=2048, help='actual input size')

    parser.add_argument('--sup_scale', type=float or int, default=1, help='scale factor for labeled clients when computing model distance')
    parser.add_argument('--dist_scale', type=float or int, default=1e4,
                        help='scale factor when computing model distance')
    parser.add_argument('--input_sz', type=int, default=32, help='actual input size')
    parser.add_argument('--pre_sz', type=int, default=40, help='image size for pre-processing')
    parser.add_argument('--unsup_num', type=int, default=9, help='number of unsupervised clients')
    parser.add_argument('--cos', action='store_true', help='resume from checkpoint')
    parser.add_argument('--EVAL', type=int, default=20,
                        help='evaluation used for training')

    parser.add_argument('--un_dist',default='',type=str,choices=["avg", "prev","mix"], help='resume from checkpoint')
    parser.add_argument('--un_dist_onlyunsup', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    args.time_current = str(int(time.time()))
    args.tensorboard_path = os.path.join('tensorboard', args.dataset, args.time_current)
    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)
    args.snapshot_path = os.path.join('model', args.dataset)
    if not os.path.isdir(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    return args
