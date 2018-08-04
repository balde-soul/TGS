# coding=utf-8
import pandas as pd
import Putil.data.split as spl
import numpy as np
from colorama import Fore
from optparse import OptionParser
import os
from openpyxl import load_workbook

parser = OptionParser(usage="usage:%prog [options] arg1 arg2")
project_data_root = '../Data/tgs/'

parser.add_option(
    '--cross_amount',
    action='store',
    type=int,
    dest='CrossAmount',
    default=10,
    help='special the max cross number for the split ,'
         'default is 10'
)
parser.add_option(
    '--info_file',
    action='store',
    type=str,
    dest='InfoFile',
    default='train-info/ana-data_set.xlsx',
    help='specify the analysis result file, which combine with the project_data_root to generate the full path'
         'defaul: train-info/ana-data_set.xlsx'
)


def split(info_file, cv_amount, **options):
    data_set = os.path.join(project_data_root, info_file)

    book = load_workbook(data_set)
    writer = pd.ExcelWriter(data_set, engine='openpyxl')
    writer.book = book
    data_dict = __split(data_set, cv_amount)
    pd.DataFrame(data_dict).to_excel(writer, sheet_name='train_val_split')
    writer.save()
    pass


def __split(data_set, cv_amount):
    data = pd.read_excel(data_set, sheetname='data')
    summary = pd.read_excel(data_set, sheetname='summary')
    split_writer = pd.ExcelWriter(data_set)
    # data_root = summary['data_root'][0]
    # image_symbol = summary['images'][0]
    # mask_symbol = summary['masks'][0]

    remain = __calc_remain(data, cv_amount)

    remain_data = dict()

    effective_data = dict()
    for key in remain.keys():
        effective_data[key] = data[key][0: remain[key]['effective_value_amount']]
        pass

    # sample the remain elements
    for key in remain.keys():
        remain_data[key] = effective_data[key].sample(remain[key]['remain'])
        pass

    # remove the remain elements
    remain_for_split = dict()
    for key in remain.keys():
        remain_for_split[key] = effective_data[key][(effective_data[key].isin(remain_data[key])).replace(False, 2.0) == 2.0]
        pass

    # get the generator
    gen = dict()
    for key in remain_for_split.keys():
        gen[key] = spl.CrossSplit(np.array(remain_for_split[key])).gen_mutual_exclusion(n_cross=cv_amount)
        pass

    # generate data for every cv
    splited_data = dict()
    for key in gen.keys():
        print(Fore.GREEN + 'go into split {0}'.format(key))
        splited_data[key] = list()
        while True:
            _data, total = gen[key].__next__()
            data_str = ''
            for data_got in _data[0]:
                data_str += ('||' + data_got)
                pass
            splited_data[key].append(data_str[2:])
            print('>>total')
            if total:
                break
                pass
            else:
                pass
            pass
        remain_data_str = ''
        for element_remain in remain_data[key]:
            remain_data_str += ('||' + element_remain)
            pass
        splited_data[key].append(remain_data_str[2:])
        pass
    return splited_data
    pass


# : for all type in data calculate the remain value and the effective_val_amount
# : reamin = {
#   key in data: {
#       'effective_value_amount': store the effective value amount,
#       'remain': the remain amount
#       }
#   }
def __calc_remain(data, cv_amount):
    remain = dict()
    for label in data.keys():
        remain[label] = dict()
        element_size = data[label][data[label].notnull()].size
        remain[label]['effective_value_amount'] = element_size
        _remain = element_size % cv_amount
        remain[label]['remain'] = _remain
    return remain


if __name__ == '__main__':
    (options, args) = parser.parse_args()
    split(options.InfoFile, options.CrossAmount)
    pass
