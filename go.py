# coding=utf-8
import Putil.loger as plog
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")
from optparse import OptionParser
parser = OptionParser(usage='usage %prog [options] arg1 arg2')
project_data_root = '../Data/tgs/'
train_config_default = 'train_config.json'
data_confiog_default = 'data.json'
parser.add_option(
    '--train_config',
    action='store',
    dest='TrainConfig',
    type=str,
    default=train_config_default,
    help='specify the train configure file path relative to'
         'project_data_root: {0}'
         'default: {1}'.format(project_data_root, train_config_default)
)
parser.add_option(
    '--data_config',
    action='store',
    dest='DataConfig',
    type=str,
    default=data_confiog_default,
    help='specify the data configure file path relative to '
         'project_data_root: {0}'
         'default: {1}'.format(project_data_root, data_confiog_default)
)
epoch_default = 100
parser.add_option(
    '--eppoch',
    action='store',
    dest='Epoch',
    type=int,
    default=epoch_default,
    help='specify the epoch to train'
         'default: {0}'.format(epoch_default)
)
batch_default = 32
parser.add_option(
    '--batch',
    action='store',
    dest='Batch',
    type=int,
    default=batch_default,
    help='specify the batch to train'
         'default: {0}'.format(batch_default)
)
val_epoch_default = 3
parser.add_option(
    '--val_epoch',
    action='store',
    dest='ValEpoch',
    type=int,
    default=val_epoch_default,
    help='specify the epoch step to val'
         'default: {0}'.format(val_epoch_default)
)
save_path_symbol_default = 'save'
parser.add_option(
    '--save_path_symbol',
    action='store',
    dest='SavePathSymbol',
    type=str,
    default=save_path_symbol_default,
    help='specify the save path symbol,'
         'which combine to :project_data_root/symbol/train_config_name-data_config_name-epoch-batch-val_epoch/model]'
         'default: {0}'.format(save_path_symbol_default)
)
level_default = 'Debug'
parser.add_option(
    '--level',
    action='store',
    dest='Level',
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
import Model
import os
import json
import Data
import Putil.train_common as ptc

logger_main = plog.PutilLogConfig('main').logger()
logger_main.setLevel(plog.DEBUG)


if __name__ == '__main__':
    train_config_file_path = os.path.join(project_data_root, options.TrainConfig)
    with open(train_config_file_path, 'r') as fp:
        param = json.loads(fp.read())
        fp.close()
    model = Model.Model(param)
    data_config_file = os.path.join(project_data_root, options.DataConfig)
    with open(data_config_file, 'r') as fp:
        param_d = json.loads(fp.read())
        fp.close()
        pass
    logger_main.info(param)
    data = Data.DataConfig(param_d)
    Trainer = ptc.TrainCommon()
    save_symbol = options.SavePathSymbol
    save_path = os.path.join(os.path.join(project_data_root, save_symbol), '{0}-{1}-{2}-{3}-{4}'.format(
        os.path.split(options.TrainConfig)[-1].split('.')[0],
        os.path.split(options.DataConfig)[-1].split('.')[0],
        options.Epoch,
        options.Batch,
        options.ValEpoch
    ))
    if os.path.exists(os.path.split(save_path)[0]):
        pass
    else:
        os.makedirs(os.path.split(save_path)[0])
        pass
    Trainer.model_cv(
        model,
        data['cv_gen'],
        data['i2d'],
        options.Epoch,
        options.ValEpoch,
        options.Batch,
        save_path)
    pass
