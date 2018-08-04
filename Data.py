# coding=utf-8
import pandas as pd
import numpy as np
import os
import cv2
import Putil.np.util as npu
from colorama import Fore
import Putil.tf.build_function as ptfb
import Putil.loger as plog
import sys

root_logger = plog.PutilLogConfig('tgs/data').logger()
root_logger.setLevel(plog.DEBUG)
IndexToDataLogger = root_logger.getChild("IndexToData")
IndexToDataLogger.setLevel(plog.DEBUG)


class CvGenerator:
    def __init__(self, split_data, sheet_name):
        """
        :param split_data:
        :param sheet_name
        """
        self._split_data = pd.read_excel(split_data, sheetname=sheet_name)
        self._cross_num = self._split_data.shape[0] - 1
        pass

    def Generate(self):
        for i in range(0, self._cross_num):
            test = list()
            train = list()
            # one remain other train(except) resident test
            for j in range(0, self._split_data.shape[0] - 2):
                if i == j:
                    for k in self._split_data.loc[j]:
                        if k is not np.nan:
                            [test.append(l) for l in k.split('||')]
                            pass
                        else:
                            pass
                        pass
                    pass
                else:
                    for k in self._split_data.loc[j]:
                        if k is not np.nan:
                            [train.append(l) for l in k.split('||')]
                            pass
                        else:
                            pass
                        pass
                    pass
                pass
            for k in self._split_data.loc[self._split_data.index[-2]]:
                if k is not np.nan:
                    [test.append(l) for l in k.split('||')]
                    pass
                else:
                    pass
                pass
            for k in self._split_data.loc[self._split_data.index[-1]]:
                if k is not np.nan:
                    [train.append(l) for l in k.split('||')]
                    pass
                else:
                    pass
                pass
            train_generate = _index_generator(train).data_generate()
            test_generate = _index_generator(test).data_generate()
            yield {'train': train_generate, 'val': test_generate}
            pass
        pass
    pass


# do: use train and val collection, generate xml or image file path, unlimited , has one epoch done symbol
class _index_generator:
    def __init__(self, data):
        """
        :param data: train or test collection list
        """
        self._data = data
        self._total = False     # one epoch done symbol
        self._index = list(range(0, len(self._data)))

    # do: re shuffle the queue, and set the one epoch done symbol to false
    def _reset(self):
        self._total = False
        np.random.shuffle(self._index)
        pass

    @property
    def Total(self):
        return self._total

    # do: generate signal data
    def data_generate(self):
        while True:
            self._reset()
            for i in self._index:
                if i == self._index[-1]:
                    self._total = True
                yield {'data': self._data[i], 'total': self.Total}
                pass
            pass
        pass
    pass


index_logger = root_logger.getChild('index_to_data')
index_logger.setLevel(plog.DEBUG)


class index_to_data:
    def __init__(self, data_root, image_symbol, mask_symbol, class_num, _dtype):
        self._data_list_name = ['image', 'GT']
        self._data_root = data_root
        self._image_symbol = image_symbol
        self._mask_symbol = mask_symbol
        self._class_amount = class_num
        self._dtype = _dtype
        index_logger.info('Generate data with keys: {0}'.format(
            self._data_list_name
        ))
        pass

    @property
    def DataListName(self):
        return self._data_list_name
        pass

    @staticmethod
    def __image_process(image):
        # if len(image.shape) == 2:
        #     image = np.expand_dims(image, -1)
        #     pass
        # else:
        #     assert image.shape[-1] == 1, \
        #         IndexToDataLogger.error(
        #             Fore.RED +
        #             'image shape should be [?, ?, 1], but {0}'.format(
        #                 image.shape))
        image = np.expand_dims(cv2.resize(image, (128, 128)), -1)
        return image
        pass

    @staticmethod
    def __mask_process(mask):
        mask = cv2.resize(mask, (128, 128))
        half = (mask.max() + mask.min()) * 0.5
        mask[mask < half] = 0
        mask[mask > half] = 1
        mask = np.array(mask, np.uint8)
        return mask

    def index_to_data(self, index):
        image_path = os.path.join(
            os.path.join(self._data_root, self._image_symbol),
            index)
        mask_path = os.path.join(
            os.path.join(self._data_root, self._mask_symbol),
            index
        )
        _image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        _image = self.__image_process(_image)
        _mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        _mask = self.__mask_process(_mask)
        # one hot mask
        # _mask = np.array(_mask, npu.np_type(self._dtype))
        # _mask = _mask - _mask.min() / _mask.max()
        _mask = np.array(_mask)
        if _mask.max() != 1:
            IndexToDataLogger.warning(
                Fore.YELLOW +
                '{0}\'s max is not one convert it'.format(
                    mask_path
                ))
            _mask[_mask == _mask.max()] = 1
            _mask[_mask != _mask.max()] = 0
        try:
            mask = ptfb.dense_to_one_hot(np.expand_dims(_mask, -1), self._class_amount, npu.np_type(self._dtype).Type)
        except:
            IndexToDataLogger.error(
                'index to data processing {0} failed\n'
                'sys.exit()'.format(
                mask_path))
            import sys
            sys.exit()

        # : image change type and normal
        _i_type = np.array(_image, npu.np_type(self._dtype).Type)
        image = (_i_type - _i_type.min())/(_i_type.max() - _i_type.min())
        return {'image': image, 'GT': mask}
        pass
    pass


# generate the cv_generator list and the index_to_data list suing the param
def DataConfig(param):
    """

    :param param: dict red from data_config
    :return:
    """
    global root_logger
    logger = root_logger.getChild('DataConfig')
    cv_gen = dict()
    i2d = dict()
    train_symbol_list = param['train_root']
    data_root = param['data_root']
    standard_len = len(train_symbol_list)
    data_symbol_list = param['data']
    split_file_list = param['split_file']
    stander_keys = ['train_image', 'train_mask']
    try:
        assert False not in [i == standard_len for i in [len(data_symbol_list), len(split_file_list)]]
    except AssertionError:
        logger.error('len should be the same of : train_root, data, split_data'
                     'but are: {0}'.format(
            [len(train_symbol_list), len(data_symbol_list), len(split_file_list)]))
        sys.exit()
        pass
    try:
        index = [(False not in [j in stander_keys for j in list(i.keys())]) for i in data_symbol_list].index(False)
        logger.error('keys of element in data list should be: {0}\n'
                     'but {1} has keys: {2}'.format(
            stander_keys, index, data_symbol_list[index].keys()
        ))
        sys.exit()
        pass
    except Exception as e:
        if type(e).__name__ == 'ValueError':
            pass
        else:
            logger.error(e)
            sys.exit()
        pass

    dtype = param['dtype']
    key = 0
    for s in zip(train_symbol_list, data_symbol_list, split_file_list):
        split_file_name, split_file_sheet = s[2].split(':')
        split_data = os.path.join(data_root, split_file_name)
        image_symbol = s[1]['train_image']
        mask_symbol = s[1]['train_mask']
        cv_gen[key] = CvGenerator(split_data, split_file_sheet).Generate()
        i2d[key] = index_to_data(
            os.path.join(data_root, s[0]),
            image_symbol,
            mask_symbol,
            2,
            dtype
        )
        key += 1
        pass
    return {'cv_gen': cv_gen, 'i2d': i2d}
    pass


if __name__ == '__main__':
    cvg = CvGenerator('../Data/tgs/train-info/ana-data_set.xlsx', 'train_val_split')
    cv_gen = cvg.Generate()
    data = index_to_data('../Data/tgs/train',
                         image_symbol='images',
                         mask_symbol='masks',
                         class_num=2,
                         _dtype=0.32
                         )
    index = cv_gen.__next__()
    for i in range(0, 5):
        _data = data.index_to_data(index['train'].__next__()['data'])
        cv2.imwrite('./test/{0}.png'.format(i), _data['image'] * 255)
        cv2.imwrite('./test/{0}-n.png'.format(i), _data['label'][:, :, 0] * 255)
        cv2.imwrite('./test/{0}-p.png'.format(i), _data['label'][:, :, 1] * 255)
        # cv2.imwrite('./test/{0}_m.bmp'.format(i), _data['label'])
    pass

