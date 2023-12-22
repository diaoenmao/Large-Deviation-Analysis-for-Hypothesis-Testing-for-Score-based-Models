import argparse
import datetime
import os
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from metric import make_metric, make_logger
from model import make_model
from module import save, load, to_device, process_control, resume, make_footprint, make_ht

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    folder_path = os.path.join('output', 'result')
    output_folder_path = os.path.join('output', 'result_2')

    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        result = load(file_path)
        base_result = result['ht_state_dict']

        base_result['fpr-threshold'] = [base_result['fpr'][i]['threshold'] for i in range(len(base_result['fpr']))]
        base_result['fpr-error'] = [base_result['fpr'][i]['error'] for i in range(len(base_result['fpr']))]
        base_result['fnr-threshold'] = [base_result['fnr'][i]['threshold'] for i in range(len(base_result['fnr']))]
        base_result['fnr-error'] = [base_result['fnr'][i]['error'] for i in range(len(base_result['fnr']))]
        del base_result['threshold']
        del base_result['fpr']
        del base_result['fnr']
        result['ht_state_dict'] = base_result

        output_file_path = os.path.join(output_folder_path, filename)
        save(result, output_file_path)
    return


if __name__ == "__main__":
    main()
