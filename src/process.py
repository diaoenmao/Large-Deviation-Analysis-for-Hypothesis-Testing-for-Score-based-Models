import os
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from module import save, load, makedir_exist_ok
from collections import defaultdict
from scipy.integrate import quad

result_path = os.path.join('output', 'result')
save_format = 'png'
vis_path = os.path.join('output', 'vis', '{}'.format(save_format))
num_experiment = 1
exp = [str(x) for x in list(range(num_experiment))]
dpi = 300
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'

num_trials = 10
num_threshold = 300


def make_control(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(mode, data, model):
    if mode == 'ptb':
        if data == 'MVN':
            # test_mode = ['lrt-t', 'lrt-e', 'hst-t', 'hst-e']
            test_mode = ['lrt-e', 'hst-t', 'hst-e']
            # test_mode = ['hst-t']
            ptb = []
            ptb_mean = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9, 0.95,
                        1, 2]
            ptb_logvar = float(0)
            for i in range(len(ptb_mean)):
                ptb_mean_i = float(ptb_mean[i])
                ptb_i = '{}-{}'.format(ptb_mean_i, ptb_logvar)
                ptb.append(ptb_i)
            control_name = [[[data], [model], test_mode, ptb]]
            controls_mean = make_control(control_name)
            ptb = []
            ptb_mean = float(0)
            ptb_logvar = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9,
                          0.95, 1, 2]
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_i = '{}-{}'.format(ptb_mean, ptb_logvar_i)
                ptb.append(ptb_i)
            control_name = [[[data], [model], test_mode, ptb]]
            controls_logvar = make_control(control_name)
            controls = controls_mean + controls_logvar
        elif data == 'RBM':
            test_mode = ['hst-t', 'hst-e']
            ptb = []
            ptb_W = [0.005, 0.007, 0.009, 0.01, 0.011, 0.012, 0.014, 0.015, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035,
                     0.04, 0.045, 0.05, 0.075, 0.1]
            for i in range(len(ptb_W)):
                ptb_W_i = float(ptb_W[i])
                ptb_i = '{}'.format(ptb_W_i)
                ptb.append(ptb_i)
            control_name = [[[data], [model], test_mode, ptb]]
            controls_W = make_control(control_name)
            controls = controls_W
        elif data == 'EXP':
            test_mode = ['lrt-e', 'hst-t', 'hst-e']
            ptb = []
            ptb_tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.0]
            for i in range(len(ptb_tau)):
                ptb_W_i = float(ptb_tau[i])
                ptb_i = '{}'.format(ptb_W_i)
                ptb.append(ptb_i)
            control_name = [[[data], [model], test_mode, ptb]]
            controls_W = make_control(control_name)
            controls = controls_W
        else:
            raise ValueError('not valid data')
    elif mode == 'ds':
        if data == 'MVN':
            # test_mode = ['lrt-t', 'lrt-e', 'hst-t', 'hst-e']
            test_mode = ['lrt-e', 'hst-t', 'hst-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 100, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_mean = float(1)
            ptb_logvar = float(0)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], [model], test_mode, ptb, data_size]]
            controls_mean = make_control(control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, data_size]]
            controls_logvar = make_control(control_name)
            controls = controls_mean + controls_logvar
        elif data == 'RBM':
            test_mode = ['hst-t', 'hst-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 100, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_W = float(0.03)
            ptb = ['{}'.format(ptb_W)]
            control_name = [[[data], [model], test_mode, ptb, data_size]]
            controls_W = make_control(control_name)
            controls = controls_W
        elif data == 'EXP':
            test_mode = ['hst-t', 'hst-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 100, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_tau = float(1)
            ptb = ['{}'.format(ptb_tau)]
            control_name = [[[data], [model], test_mode, ptb, data_size]]
            controls_tau = make_control(control_name)
            controls = controls_tau
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    # mode = ['ptb', 'ds']
    data_name = ['MVN', 'RBM', 'EXP']
    mode = ['ptb']
    controls = []
    for i in range(len(mode)):
        mode_i = mode[i]
        for j in range(len(data_name)):
            data_j = data_name[j]
            model_j = data_j.lower()
            control_list = make_control_list(mode_i, data_j, model_j)
            controls = controls + control_list
    processed_result = process_result(controls)
    df_mean = make_df(processed_result, 'mean', True)
    df_history = make_df(processed_result, 'history', False)
    make_vis_threshold(df_history)
    make_vis_roc(df_history)
    return


def tree():
    return defaultdict(tree)


def process_result(controls):
    result = tree()
    for control in controls:
        model_tag = '_'.join(control)
        gather_result(list(control), model_tag, result)
    summarize_result(None, result)
    processed_result = tree()
    extract_result(processed_result, result, [])
    return processed_result


def gather_result(control, model_tag, processed_result):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for metric_name in base_result['ht_state_dict']:
                metric_name_ = 'test/{}'.format(metric_name)
                processed_result[metric_name_]['history'][exp_idx] = np.stack(
                    base_result['ht_state_dict'][metric_name], axis=0)
            for i in range(num_trials):
                threshold_i = processed_result['test/threshold']['history'][exp_idx][i]
                fpr_i = processed_result['test/fpr']['history'][exp_idx][i]
                fnr_i = processed_result['test/fnr']['history'][exp_idx][i]
                sorted_indices = np.argsort(threshold_i)
                threshold_i = threshold_i[sorted_indices]
                fpr_i = fpr_i[sorted_indices]
                fnr_i = fnr_i[sorted_indices]
                valid_mask_fpr_i = np.logical_and(fpr_i >= -1e-5, fpr_i <= 1 + 1e-5)
                valid_mask_fnr_i = np.logical_and(fnr_i >= -1e-5, fnr_i <= 1 + 1e-5)
                if valid_mask_fpr_i.sum() < len(fpr_i) * 0.8 or valid_mask_fnr_i.sum() < len(fnr_i) * 0.8:
                    break
                fpr_i = np.interp(threshold_i, threshold_i[valid_mask_fpr_i], fpr_i[valid_mask_fpr_i])
                fnr_i = np.interp(threshold_i, threshold_i[valid_mask_fnr_i], fnr_i[valid_mask_fnr_i])
                processed_result['test/threshold']['history'][exp_idx][i] = threshold_i
                processed_result['test/fpr']['history'][exp_idx][i] = fpr_i
                processed_result['test/fnr']['history'][exp_idx][i] = fnr_i
            for i in range(len(processed_result['test/fpr']['history'][exp_idx])):
                fpr_i = processed_result['test/fpr']['history'][exp_idx][i]
                fnr_i = processed_result['test/fnr']['history'][exp_idx][i]
                sorted_indices = np.argsort(fpr_i)
                fpr_i = fpr_i[sorted_indices]
                fnr_i = fnr_i[sorted_indices]
                auroc_i, _ = quad(lambda x: np.interp(x, fpr_i, 1 - fnr_i), 0, 1)
                processed_result['test/AUROC']['mean'][i] = auroc_i
            for metric_name in ['test/threshold', 'test/fpr', 'test/fnr']:
                processed_result[metric_name]['history'][exp_idx] = (
                    processed_result[metric_name]['history'][exp_idx].reshape(-1).tolist())
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        gather_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
    return


def summarize_result(key, value):
    if key in ['mean', 'history']:
        value['summary']['value'] = np.stack(list(value.values()), axis=0)
        value['summary']['mean'] = np.mean(value['summary']['value'], axis=0)
        value['summary']['std'] = np.std(value['summary']['value'], axis=0)
        value['summary']['max'] = np.max(value['summary']['value'], axis=0)
        value['summary']['min'] = np.min(value['summary']['value'], axis=0)
        value['summary']['argmax'] = np.argmax(value['summary']['value'], axis=0)
        value['summary']['argmin'] = np.argmin(value['summary']['value'], axis=0)
        value['summary']['value'] = value['summary']['value'].tolist()
    else:
        for k, v in value.items():
            summarize_result(k, v)
        return
    return


def extract_result(extracted_processed_result, processed_result, control):
    def extract(metric_name, mode):
        output = False
        if metric_name in ['test/AUROC']:
            if mode == 'mean':
                output = True
        if metric_name in ['test/threshold', 'test/fpr', 'test/fnr']:
            if mode == 'history':
                output = True
        return output

    if 'summary' in processed_result:
        control_name, metric_name, mode = control
        if not extract(metric_name, mode):
            return
        stats = ['mean', 'std']
        for stat in stats:
            exp_name = '_'.join([control_name, metric_name.split('/')[1], stat])
            extracted_processed_result[mode][exp_name] = processed_result['summary'][stat]
    else:
        for k, v in processed_result.items():
            extract_result(extracted_processed_result, v, control + [k])
    return


def make_df(processed_result, mode, write):
    df = defaultdict(list)
    for exp_name in processed_result[mode]:
        exp_name_list = exp_name.split('_')
        df_name = '_'.join([*exp_name_list])
        index_name = [1]
        df[df_name].append(pd.DataFrame(data=processed_result[mode][exp_name].reshape(1, -1), index=index_name))
    startrow = 0
    with pd.ExcelWriter(os.path.join(result_path, 'result_{}.xlsx'.format(mode)), engine='xlsxwriter') as writer:
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
            if write:
                df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1, header=False, index=False)
                writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
    return df


def make_vis_threshold(df_history):
    label_dict = {'lrt-e': 'LRT (Empirical)', 'hst-e': 'HST (Empirical)', 'lrt-t': 'LRT (Theoretical)',
                  'hst-t': 'HST (Theoretical)'}
    color_dict = {'lrt-e': 'red', 'hst-e': 'blue', 'lrt-t': 'orange', 'hst-t': 'dodgerblue'}
    linestyle_dict = {'lrt-e': '-', 'hst-e': '--', 'lrt-t': ':', 'hst-t': '-.'}
    marker_dict = {'lrt-e': 'o', 'hst-e': 's', 'lrt-t': 'p', 'hst-t': 'd'}
    loc_dict = {'fpr': 'upper right', 'fnr': 'lower right'}
    fontsize_dict = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        mask = metric_name in ['fpr', 'fnr'] and stat == 'mean'
        if mask:
            pivot = df_name_list[2]
            df_name_threshold = '_'.join([*df_name_list[:-2], 'threshold', 'mean'])
            fig_name = '_'.join([*df_name_list[:2], *df_name_list[3:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            x = df_history[df_name_threshold].iloc[0].to_numpy()
            x = x.reshape(-1, num_threshold)
            x = x.mean(axis=0)
            y = df_history[df_name].iloc[0].to_numpy()
            y = y.reshape(-1, num_threshold)
            y_mean = y.mean(axis=0)
            y_std = y.std(axis=0) / np.sqrt(y.shape[0])
            xlabel = 'Threshold'
            ylabel = metric_name.upper()
            ax_1.plot(x, y_mean, label=label_dict[pivot], color=color_dict[pivot],
                      linestyle=linestyle_dict[pivot])
            ax_1.fill_between(x, (y_mean - y_std), (y_mean + y_std), color=color_dict[pivot], alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(fontsize=fontsize_dict['legend'])
    for fig_name in fig:
        fig_name_list = fig_name.split('_')
        data_name = fig_name_list[0]
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        dir_name = 'threshold'
        dir_path = os.path.join(vis_path, dir_name, data_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig_name)
    return


def make_vis_roc(df_history):
    label_dict = {'lrt-e': 'LRT (Empirical)', 'hst-e': 'HST (Empirical)', 'lrt-t': 'LRT (Theoretical)',
                  'hst-t': 'HST (Theoretical)'}
    color_dict = {'lrt-e': 'red', 'hst-e': 'blue', 'lrt-t': 'orange', 'hst-t': 'dodgerblue'}
    linestyle_dict = {'lrt-e': '-', 'hst-e': '--', 'lrt-t': ':', 'hst-t': '-.'}
    marker_dict = {'lrt-e': 'o', 'hst-e': 's', 'lrt-t': 'p', 'hst-t': 'd'}
    loc_dict = {'fpr': 'lower right'}
    fontsize_dict = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        mask = metric_name in ['fpr'] and stat == 'mean'
        if mask:
            pivot = df_name_list[2]
            df_name_fnr = '_'.join([*df_name_list[:-2], 'fnr', 'mean'])
            fig_name = '_'.join([*df_name_list[:2], *df_name_list[3:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            x = df_history[df_name].iloc[0].to_numpy()
            x = x.reshape(-1, num_threshold)
            x = x.mean(axis=0)
            y = 1 - df_history[df_name_fnr].iloc[0].to_numpy()
            y = y.reshape(-1, num_threshold)
            y_mean = y.mean(axis=0)
            y_std = y.std(axis=0) / np.sqrt(y.shape[0])
            xlabel = 'FPR'
            ylabel = 'TPR'
            ax_1.plot(x, y_mean, label=label_dict[pivot], color=color_dict[pivot],
                      linestyle=linestyle_dict[pivot])
            ax_1.fill_between(x, (y_mean - y_std), (y_mean + y_std), color=color_dict[pivot], alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.legend(loc=loc_dict[metric_name], fontsize=fontsize_dict['legend'])
    for fig_name in fig:
        fig_name_list = fig_name.split('_')
        data_name = fig_name_list[0]
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        dir_name = 'roc'
        dir_path = os.path.join(vis_path, dir_name, data_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
