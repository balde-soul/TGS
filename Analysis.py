# coding=utf-8
from optparse import OptionParser
from colorama import Fore
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import Putil.visual.matplotlib_plot as Pplot


plot_type = Pplot.random_type()

parser = OptionParser(usage="usage:%prog [options] arg1 arg2")
data_root = '../Data/tgs/'

parser.add_option(
    '--data_symbol',
    action='store',
    dest='DataSymbol',
    default='train',
    help='specify the data in the data_root, to combine the full data path'
         'default: train'
)

parser.add_option(
    '--image_symbol',
    action='store',
    dest='ImageSymbol',
    default='images',
    help='specify the symbol of images store path, '
         'combine the full data path to generator the image folder path'
         'default: images'
)

parser.add_option(
    '--mask_symbol',
    action='store',
    dest='MaskSymbol',
    default='masks',
    help='specify the symbol of masks store path, '
         'combine the full data path to generator the image folder path'
         'default: masks'
)

parser.add_option(
    '--result_file',
    action='store',
    dest='ResultFile',
    default='train-info/ana',
    help='specify the path base on the data root'
         'to store the result file'
         'default: train-info/ana'
)


# : generate the index xslx file and check if some image leak label
def Analysis(data_name, images_symbol, mask_symbol, file_basics_path):
    """
    :param data_name: combine with the data_root , make the data_path
    :param file_basics_path:
        generate file base on file_basics_path(label-1 rate distribution png/data_set.xlsx/),
        which contain the rate_label_1, the index(0--..)
        use os.path.join(data_root ,file_basics_path) + 'label.xxx' as the file name
        such as: 'train_ana/train_origin'
    :return:
    """
    data_path = os.path.join(data_root, data_name)
    images_path = os.path.join(data_path, images_symbol)
    assert os.path.exists(images_path), print(Fore.RED + 'image path: {0} is not exit!'.format(images_path))
    masks_path = os.path.join(data_path, mask_symbol)
    assert os.path.exists(masks_path), print(Fore.RED + 'mask path: {0} is not exit!'.format(masks_path))
    save_path = os.path.join(data_root, os.path.split(file_basics_path)[0])
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
        pass

    distribution_figure = os.path.join(data_root, file_basics_path) + '-data_distribution.png'

    data_file_path = os.path.join(data_root, file_basics_path) + '-data_set.xlsx'
    data = dict()   # to data_set: data
    data['only-background'] = list()
    data['with-foreground'] = list()

    data_summary = dict()   # to data_set:summary
    data_summary['lack_mask'] = list()
    data_summary['mask_image_shape_not_match'] = list()
    data_summary['mask_needless'] = list()
    data_summary['data_root'] = [data_path]
    data_summary['images'] = [images_symbol]
    data_summary['masks'] = [mask_symbol]

    distribution = list()  # to image
    analysised_count = 0
    print(Fore.GREEN + 'starting analysis')
    for image_name in os.listdir(images_path):
        analysised_count += 1
        if analysised_count % 1000 == 0:
            print(Fore.GREEN + 'successful analysis {0}'.format(analysised_count))
        image_path = os.path.join(images_path, image_name)
        # :check mask exit?
        mask_path = __check_mask_exit(image_name, masks_path)
        if mask_path is not None:
            # read the mask and image
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            # : check mask vs image match
            if __check_image_mask_match(image, mask) is True:
                pass
            else:
                data_summary['mask_image_shape_not_match'].append(image_name)
                pass
            # : get not zero amount
            amount = __calc_p_label_rate(mask)
            if amount == 0:
                data['only-background'].append(image_name)
            else:
                data['with-foreground'].append(image_name)
            distribution.append(amount)
        else:
            data_summary['lack_mask'].append(image_name)
        pass
    analysised_count = 0
    for mask_name in os.listdir(masks_path):
        analysised_count += 1
        if analysised_count % 1000 == 0:
            print(Fore.GREEN + 'successful analysis mask {0}'.format(analysised_count))
        if __check_mask_useful(mask_name, images_path) is True:
            pass
        else:
            data_summary['mask_needless'].append(mask_name)
            pass
        pass
    print(Fore.GREEN + 'fitting len for data')
    __fit_len_to_pd(data)
    print(Fore.GREEN + 'fitting len for summary')
    __fit_len_to_pd(data_summary)
    print(Fore.GREEN + 'data to DataFrame')
    data_frame = pd.DataFrame(data)
    print(Fore.GREEN + 'summary to DataFrame')
    summary_frame = pd.DataFrame(data_summary)
    writer = pd.ExcelWriter(data_file_path)
    print(Fore.GREEN + 'writing DataFrame')
    data_frame.to_excel(writer, 'data')
    print(Fore.GREEN + 'writing summary')
    summary_frame.to_excel(writer, 'summary')
    print(Fore.GREEN + 'distribution DataFrame')
    # distribution_fram.to_excel(writer, 'distribution')
    writer.save()
    print(Fore.GREEN + 'generating distribution figure')
    plt.hist(distribution, 'auto', density=True,  histtype="bar", facecolor='g', alpha=0.9)
    plt.savefig(distribution_figure)
    pass


def __check_mask_exit(image_name, mask_path):
    mask_path = os.path.join(mask_path, image_name)
    if os.path.exists(mask_path):
        return mask_path
    else:
        return None
    pass


def __check_mask_useful(mask_name, image_path):
    image_path = os.path.join(image_path, mask_name)
    if os.path.exists(image_path):
        return True
    else:
        return False
    pass


def __check_image_mask_match(image, mask):
    # : shape
    if image.shape[0: 2] != mask.shape[0: 2]:
        return False
        pass
    return True
    pass


def __calc_p_label_rate(mask):
    try:
        channel = mask.shape[2]
    except IndexError as e:
        channel = 1
        pass
    not_zero_amount = len(np.nonzero(mask)[0]) / channel
    return int(not_zero_amount)
    pass


def __fit_len_to_pd(dictionary):
    len_list = []
    for member in dictionary.keys():
        if type(dictionary[member]).__name__ == 'list':
            len_list.append(len(dictionary[member]))
            pass
        else:
            pass
        pass
    max_len = max(len_list)
    for member in dictionary.keys():
        if type(dictionary[member]).__name__ == 'list':
            dictionary[member] = dictionary[member] + [None] * (max_len - len(dictionary[member]))
            pass
        else:
            pass
        pass
    pass


if __name__ == '__main__':
    (options, args) = parser.parse_args()
    Analysis(options.DataSymbol, options.ImageSymbol, options.MaskSymbol, options.ResultFile)
    pass
pass
