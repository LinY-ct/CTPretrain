import sys
sys.path.append('..')
import argparse
import os
import random
import numpy as np
import torch
import importlib
import yaml
import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）

def update_parser_defaults(opt, config):
    for k, v in config.items():
        if k in vars(opt):
            setattr(opt, k, v)
    return opt
def get_parser():
    parser = argparse.ArgumentParser(description='CT-Recon Main')
    parser.add_argument('--config', default='', type=str, help='config.yaml path')
    parser.add_argument('--trainer_mode', default='train', type=str, help='train or test')
    parser.add_argument('--network', default='', type=str, help='networkname')
    parser.add_argument('--net_dict', default="", type=str, help='string of dict containing network arguments')
    parser.add_argument('--PretrainWeight',
                        default="",
                        type=str, help='PretrainWeightPath')

    # Tensorboard
    parser.add_argument('--checkpoint_dir', type=str, default='default', help='detail folder of checkpoint')
    parser.add_argument('--tensorboard_dir', type=str, default='default', help='detail folder of tensorboard')
    parser.add_argument('--checkpoint_root', type=str, default='' ,
                        help='where to save the checkpoint')
    parser.add_argument('--use_tensorboard', action='store_true', default=True, help='whether to use tensorboard')
    parser.add_argument('--tensorboard_root', type=str, default="",
                        help='root path of tensorboard, project path')
    parser.add_argument('--log_interval', type=int, default=400, help='logging interval by iteration')
    parser.add_argument('--use_tqdm', action='store_true', default=True, help='whether to use tqdm')
    # Resume
    parser.add_argument('--resume', default=False, action='store_true',
                        help='resume network training or not, load network param')
    parser.add_argument('--tune', default=False, action='store_true',
                        help='resume network training or not, load network param')
    parser.add_argument('--net_checkpath', default="", type=str, help='network checkpoint path')
    parser.add_argument('--resume_opt', default=False, action='store_true',
                        help='resume optimizer or not, load opt param')
    parser.add_argument('--opt_checkpath', default="", type=str, help='optimizer checkpath')
    # data_path
    parser.add_argument('--dataset_path', type=str, default="/MayoData/npy/", help='dataset path')
    parser.add_argument('--dataset_name', default='aapm', type=str,
                        help='which dataset, size640,size320,deepleision.etc.')
    parser.add_argument('--dataset_shape', type=int, default=256, help='modify shape in dataset')
    parser.add_argument('--num_train', default=None, type=int, help='number of training examples')
    parser.add_argument('--num_val', default=None, type=int, help='number of validation examples')
    # dataloader
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--shuffle', default=True, type=bool, help='dataloader shuffle, False if test and val')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers, 4 is a good choice')
    parser.add_argument('--drop_last', default=False, type=bool, help='dataloader droplast')
    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta2')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--save_epochs', default=1, type=int)
    # scheduler
    parser.add_argument('--scheduler', default='step', type=str, help='name of the scheduler')
    parser.add_argument('--step_size', default=10, type=int, help='step size for StepLR')
    parser.add_argument('--milestones', nargs='+', type=int, help='milestones for MultiStepLR')
    parser.add_argument('--step_gamma', default=0.5, type=float, help='learning rate reduction factor')
    parser.add_argument('--poly_iters', default=10, type=int,
                        help='the number of steps that the scheduler decays the learning rate')
    parser.add_argument('--poly_power', default=2, type=float, help='the power of the polynomial')
    # tester args
    parser.add_argument('--tester_save_name', default='default', type=str, help='name of test')
    parser.add_argument('--tester_save_image', default=True, action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--tester_save_path', default='', type=str,
                        help='path for saving tester result')
    parser.add_argument('--tester_save_matnum', default=10, type=int, help="number of save mat file")
    # degenetaion
    # parser.add_argument('--degenetaion_type', default=[], type=list, help="degenetaion type of ct")
    # parser.add_argument('--dose_range', default=[], type=list, help="low dose range of LDCT")
    # parser.add_argument('--views_range', default=[], type=list, help="view range of SVCT")
    # parser.add_argument('--lv_views_range', default=[], type=list, help="view range of LVCT")
    parser.add_argument('--degenetaion_type', nargs='+', type=str,default=['ld', 'sv', 'lv','ld_sv','ld_lv'], help="degenetaion type of ct")
    parser.add_argument('--dose_range', nargs='+', type=int,default=[1000000, 500000, 100000,50000, 10000], help="low dose range of LDCT")
    parser.add_argument('--views_range', nargs='+', type=int, default=[240, 180, 144, 120, 90, 80, 72, 60, 48, 36, 18],help="view range of SVCT")
    parser.add_argument('--lv_views_range', nargs='+', type=int,default=[360, 240, 180, 120], help="view range of LVCT")

    parser.add_argument('--de_type', type=str, default='all')
    parser.add_argument('--dose', type=int, default=1)
    parser.add_argument('--view', type=int, default=720)


    # PIP Parameter
    parser.add_argument('--high_reg_loss', default='angle', type=str,
                        help='option of DDL loss, reg for high level prompt, [angle, cosine, None]')

    return parser

def load_class(path, net_name,*args, **kwargs): # 类存放的文件为class_name.py
    module = importlib.import_module(path + '.' + net_name) # 导入模块
    return getattr(module, net_name)(*args, **kwargs) # 获取类

def main(opt):

    net = None

    if opt.network == 'AdaIR':
        print('load model : ' + 'AdaIR')
        net_module = importlib.import_module('models' + '.' + 'AdaIR')
        net = getattr(net_module, 'AdaIR')()  # 获取类
    elif opt.network == 'freeseed':
        print('load model : ' + 'freeseed')
        net_module = importlib.import_module('models' + '.' + 'freeseed')
        net1 = getattr(net_module, 'FreeNet')()
        net2 = getattr(net_module, 'SeedNet')()
    elif opt.network == 'PIPNet_Restormer_onskip_inter':
        print('load model : ' + 'PIPNet_Restormer_onskip_inter')
        net_module = importlib.import_module('models' + '.' + 'PIP_Net')
        net = getattr(net_module, 'PIPNet_Restormer_onskip_inter')(
            inp_channels=3, out_channels=3,
            decoder=True,
            use_detask_label=True,
            use_detask_prompt=True,
            use_CLIP_prompt=False,
            use_SAM_prompt=False,
            use_degradation_sensor=False,
            high_prompt_dim=1,
            low_prompt_dims=[64, 128, 256] ,
            prompt_interaction_mode='pip_cross_topm',
            degradation_num=5,
            low_prompt_sizes=[64, 32, 16],
        )
    elif opt.network == 'PIPNet_Restormer_onskip_interV2':
        print('load model : ' + 'PIPNet_Restormer_onskip_interV2')
        net_module = importlib.import_module('models' + '.' + 'PIP_Net')
        net = getattr(net_module, 'PIPNet_Restormer_onskip_interV2')(
            inp_channels=3, out_channels=3,
            decoder=True,
            use_detask_label=True,
            use_detask_prompt=True,
            use_CLIP_prompt=False,
            use_SAM_prompt=False,
            use_degradation_sensor=False,
            high_prompt_dim=1,
            low_prompt_dims=[64, 128, 256],
            prompt_interaction_mode='pip_cross_topm',
            degradation_num=5,
            low_prompt_sizes=[64, 32, 16],
        )
    elif opt.network == 'PIPNet_Restormer_onskip_interV3':
        print('load model : ' + 'PIPNet_Restormer_onskip_interV3')
        net_module = importlib.import_module('models' + '.' + 'PIP_Net')
        net = getattr(net_module, 'PIPNet_Restormer_onskip_interV3')(
            inp_channels=3, out_channels=3,
            decoder=True,
            use_detask_label=True,
            use_detask_prompt=True,
            use_CLIP_prompt=False,
            use_SAM_prompt=False,
            use_degradation_sensor=False,
            high_prompt_dim=1,
            low_prompt_dims=[64, 128, 256],
            prompt_interaction_mode='pip_cross_topm',
            degradation_num=5,
            low_prompt_sizes=[64, 32, 16],
        )
    elif opt.network == 'PIPNet_Restormer_onskip_interV4':
        print('load model : ' + 'PIPNet_Restormer_onskip_interV4')
        net_module = importlib.import_module('models' + '.' + 'PIP_Net')
        net = getattr(net_module, 'PIPNet_Restormer_onskip_interV4')(
            inp_channels=3, out_channels=3,
            decoder=True,
            use_detask_label=True,
            use_detask_prompt=True,
            use_CLIP_prompt=False,
            use_SAM_prompt=False,
            use_degradation_sensor=False,
            high_prompt_dim=1,
            low_prompt_dims=[64, 128, 256],
            prompt_interaction_mode='pip_cross_topm',
            degradation_num=5,
            low_prompt_sizes=[64, 32, 16],
        )
    elif opt.network == 'PIPNet_Restormer_onskip_interV5':
        print('load model : ' + 'PIPNet_Restormer_onskip_interV5')
        net_module = importlib.import_module('models' + '.' + 'PIP_Net')
        net = getattr(net_module, 'PIPNet_Restormer_onskip_interV5')(
            inp_channels=3, out_channels=3,
            decoder=True,
            use_detask_label=True,
            use_detask_prompt=True,
            use_CLIP_prompt=False,
            use_SAM_prompt=False,
            use_degradation_sensor=False,
            high_prompt_dim=1,
            low_prompt_dims=[64, 128, 256],
            prompt_interaction_mode='pip_cross_topm',
            degradation_num=5,
            low_prompt_sizes=[64, 32, 16],
        )
    elif opt.network in ['Restormer', 'Restormer_CL', 'Restormer_GGB3CL','Restormer_GGB3CLAndCL']:
        print('load model: ' + 'Restormer')
        net_module = importlib.import_module('models' + '.' + 'AdaIR')
        net = getattr(net_module, 'AdaIR')(decoder=False)  # 获取类
    elif opt.network == 'Restormer_GGB':
        print('load model: ' + 'Restormer_GGB')
        net_module = importlib.import_module('models' + '.' + 'Restormer_GGB')
        net = getattr(net_module, 'Restormer_GGB')()  # 获取类
    elif opt.network == 'Restormer_GGB2':
        print('load model: ' + 'Restormer_GGB2')
        net_module = importlib.import_module('models' + '.' + 'Restormer_GGB2')
        net = getattr(net_module, 'Restormer_GGB2')()  # 获取类
    elif opt.network in ['Restormer_GGB3', 'Restormer_GGB3_CL']:
        print('load model: ' + 'Restormer_GGB3')
        net_module = importlib.import_module('models' + '.' + 'Restormer_GGB3')
        net = getattr(net_module, 'Restormer_GGB3')()  # 获取类
    elif opt.network  in ['Restormer_GGB3_Encoder', 'Restormer_GGB3_Encoder_CL']:
        print('load model: ' + 'Restormer_GGB3_Encoder')
        net_module = importlib.import_module('models' + '.' + 'Restormer_GGB3_Encoder')
        net = getattr(net_module, 'Restormer_GGB3_Encoder')()
    elif opt.network == 'Restormer_GGB3_Encoder_Decoder':
        print('load model: ' + 'Restormer_GGB3_Encoder_Decoder')
        net_module = importlib.import_module('models' + '.' + 'Restormer_GGB3_Encoder_Decoder')
        net = getattr(net_module, 'Restormer_GGB3_Encoder_Decoder')()
    elif opt.network == 'Restormer_GGB3_Encoder_Decoder_v2':
        print('load model: ' + 'Restormer_GGB3_Encoder_Decoder_v2')
        net_module = importlib.import_module('models' + '.' + 'Restormer_GGB3_Encoder_Decoder_v2')
        net = getattr(net_module, 'Restormer_GGB3_Encoder_Decoder_v2')()
    elif opt.network == 'Restormer_GGB3_Encoder_GGB3CLv2':
        print('load model: ' + 'Restormer_GGB3_Encoder_GGB3CLv2')
        net_module = importlib.import_module('models' + '.' + 'Restormer_GGB3_Encoder_GGB3CLv2')
        net = getattr(net_module, 'Restormer_GGB3_Encoder_GGB3CLv2')()
    elif opt.network == 'DuDoRestormer_GGB3_Encoder_GGBCLv2':
        print('load model: ' + 'DuDoRestormer_GGB3_Encoder_GGBCLv2')
        net_module = importlib.import_module('models' + '.' + 'DuDoRestormer_GGB3_Encoder_GGBCLv2')
        net = getattr(net_module, 'DuDoRestormer_GGB3_Encoder_GGBCLv2')()
    else:
        print('load model: ' + opt.network)
        net_module = importlib.import_module('models' + '.' + opt.network)
        net = getattr(net_module, opt.network)()



    if opt.trainer_mode == 'train':
        trainer = None
        if opt.dataset_name == 'MAR' :
            print('load trainer : ' + 'mar_trainer')
            trainer = load_class('trainers', 'mar_trainer', opt=opt, net=net)
            trainer.fit()
        elif opt.network == 'freeseed':
            print('load trainer : ' + 'trainer_freeseed')
            trainer = load_class('trainers', 'trainer_freeseed', opt=opt, net1=net1, net2=net2)
            trainer.fit()
        elif opt.network == 'dudofreeseed':
            print('load trainer : ' + 'trainer_dudofreeseed')
            trainer = load_class('trainers', 'trainer_dudofreeseed', opt=opt, net=net)
            trainer.fit()
        elif opt.network == 'dudotrans':
            print('load trainer : ' + 'trainer_dudotrans')
            trainer = load_class('trainers', 'trainer_dudotrans', opt=opt, net=net)
            trainer.fit()
        elif opt.network in ['Restormer_GGB3_Encoder_Decoder_v2', 'Restormer_GGB3_Encoder_GGB3CLv2']:
            print('load trainer : ' + 'trainer_GGB3CL_v2')
            trainer = load_class('trainers', 'trainer_GGB3CL_v2', opt=opt, net=net)
            trainer.fit()
        elif 'DuDoRestormer_GGB3_Encoder_GGBCLv2' == opt.network:
            print('load trainer : ' + 'DuDotrainer')
            trainer = load_class('trainers', 'DuDotrainer', opt=opt, net=net)
            trainer.fit()
        elif "_GGB3CLAndCL" in  opt.network:
            print('load trainer : ' + 'trainer_GGB3CLAndCL')
            trainer = load_class('trainers', 'trainer_GGB3CLAndCL', opt=opt, net=net)
            trainer.fit()
        elif "_CL" in  opt.network:
            print('load trainer : ' + 'trainer_CL')
            trainer = load_class('trainers', 'trainer_CL', opt=opt, net=net)
            trainer.fit()
        elif "_GGB3CL" in  opt.network:
            print('load trainer : ' + 'trainer_GGB3CL')
            trainer = load_class('trainers', 'trainer_GGB3CL', opt=opt, net=net)
            trainer.fit()
        elif "AMIR" == opt.network:
            print('load trainer : ' + 'trainer_AMIR')
            trainer = load_class('trainers', 'trainer_AMIR', opt=opt, net=net)
            trainer.fit()
        elif opt.network in ['PIPNet_Restormer_onskip_interV3', 'PIPNet_Restormer_onskip_interV4', 'PIPNet_Restormer_onskip_interV5' ] :
            print('load trainer : ' + 'trainer_PIP_V3')
            trainer = load_class('trainers', 'trainer_PIP_V3', opt=opt, net=net)
            trainer.fit()
        elif 'PIPNet_Restormer_onskip_inter' in opt.network :
            print('load trainer : ' + 'trainer_PIP')
            trainer = load_class('trainers', 'trainer_PIP', opt=opt, net=net)
            trainer.fit()
        elif 'MobileNetV2' == opt.network:
            print('load trainer : ' + 'trainer_MobileNetV2')
            trainer = load_class('trainers', 'trainer_MobileNetV2', opt=opt, net=net)
            trainer.fit()
        elif 'dinov2' ==  opt.network:
            print('load trainer : ' + 'trainer_dinov2')
            trainer = load_class('trainers', 'trainer_dinov2', opt=opt, net=net)
            trainer.fit()
        # elif opt.network in ['Restore_RWKV', 'Restore_RWKV_GGB3_Encoder'] :
        #     print('load trainer : ' + 'trainer_Restore_RWKV')
        #     trainer = load_class('trainers', 'trainer_Restore_RWKV', opt=opt, net=net)
        #     trainer.fit()
        else:
            print('load trainer : ' + 'trainer')
            trainer = load_class('trainers',  'trainer', opt=opt, net=net)
            trainer.fit()
    elif opt.trainer_mode == 'test':
        if 'PIPNet_Restormer_onskip_inter' in opt.network :
            print('load tester: ' + 'tester_PIP')
            tester = load_class('trainers', 'tester_PIP', opt=opt, net=net)
            tester.run()
        elif 'DuDoRestormer_GGB3_Encoder_GGBCLv2' == opt.network:
            tester = load_class('trainers', 'dudotester', opt=opt,net=net)
            tester.run()
        elif opt.dataset_name == 'MAR' :
            print('load tester: ' + 'mar_tester')
            tester = load_class('trainers', 'mar_tester', opt=opt, net=net)
            tester.run()
        elif opt.network == 'dudotrans':
            print('load tester: ' + 'tester_dudotrans')
            tester = load_class('trainers', 'tester_dudotrans', opt=opt,net=net)
            tester.run()
        elif opt.network == 'freeseed':
            tester = load_class('trainers', 'tester', opt=opt,net=net1)
            tester.run()
        elif opt.network == 'dudofreeseed':
            print('load tester: ' + 'tester_dudofreeseed')
            tester = load_class('trainers', 'tester_dudofreeseed', opt=opt, net=net)
            tester.run()
        else:
            tester = load_class('trainers', 'tester', opt=opt,net=net)
            tester.run()
    print('finish')



if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    # 如果提供了配置文件路径，则加载配置文件并更新参数解析器的默认值
    if opt.config:
        with open(opt.config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                opt = update_parser_defaults(opt, config)
            except yaml.YAMLError as exc:
                print(exc)

    seed = 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(opt)
