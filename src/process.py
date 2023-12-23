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
            test_mode_t = ['lrt-t', 'hst-t']
            test_mode_e = ['lrt-e', 'hst-e']
            n_t = ['1']
            n_e = ['1', '2', '4', '8', '16', '32', '64', '128']
            ptb = []
            ptb_mean = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035,
                        0.04, 0.045, 0.05, 0.075, 0.1]
            ptb_logvar = float(0)
            for i in range(len(ptb_mean)):
                ptb_mean_i = float(ptb_mean[i])
                ptb_i = '{}-{}'.format(ptb_mean_i, ptb_logvar)
                ptb.append(ptb_i)
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t]]
            controls_mean_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_mean_e = make_control(control_name_e)
            ptb = []
            ptb_mean = float(0)
            ptb_logvar = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035,
                          0.04, 0.045, 0.05, 0.075, 0.1]
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_i = '{}-{}'.format(ptb_mean, ptb_logvar_i)
                ptb.append(ptb_i)
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t]]
            controls_logvar_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_logvar_e = make_control(control_name_e)
            controls = controls_mean_t + controls_logvar_t + controls_mean_e + controls_logvar_e
        elif data == 'RBM':
            test_mode_t = ['hst-t']
            test_mode_e = ['hst-e']
            n_t = ['1']
            n_e = ['1', '2', '4', '8', '16', '32', '64', '128']
            ptb = []
            ptb_W = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035,
                     0.04, 0.045, 0.05, 0.075, 0.1]
            for i in range(len(ptb_W)):
                ptb_W_i = float(ptb_W[i])
                ptb_i = '{}'.format(ptb_W_i)
                ptb.append(ptb_i)
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t]]
            controls_W_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_W_e = make_control(control_name_e)
            controls = controls_W_t + controls_W_e
        elif data == 'EXP':
            test_mode_t = ['lrt-t', 'hst-t']
            test_mode_e = ['lrt-e', 'hst-e']
            n_t = ['1']
            n_e = ['1', '2', '4', '8', '16', '32', '64', '128']
            ptb = []
            ptb_tau = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035,
                       0.04, 0.045, 0.05, 0.075, 0.1]
            for i in range(len(ptb_tau)):
                ptb_tau_i = float(ptb_tau[i])
                ptb_i = '{}'.format(ptb_tau_i)
                ptb.append(ptb_i)
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t]]
            controls_tau_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_tau_e = make_control(control_name_e)
            controls = controls_tau_t + controls_tau_e
        else:
            raise ValueError('not valid data')
    elif mode == 'ds':
        if data == 'MVN':
            test_mode_t = ['lrt-t', 'hst-t']
            test_mode_e = ['lrt-e', 'hst-e']
            n_t = ['1']
            n_e = ['1', '2', '4', '8', '16', '32', '64', '128']
            data_size = [5, 10, 20, 40, 60, 80, 100, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_mean = float(0.01)
            ptb_logvar = float(0)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t, data_size]]
            controls_mean_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_mean_e = make_control(control_name_e)
            ptb_mean = float(0)
            ptb_logvar = float(0.01)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t, data_size]]
            controls_logvar_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_logvar_e = make_control(control_name_e)
            controls = controls_mean_t + controls_logvar_t + controls_mean_e + controls_logvar_e
        elif data == 'RBM':
            test_mode_t = ['hst-t']
            test_mode_e = ['hst-e']
            n_t = ['1']
            n_e = ['1', '2', '4', '8', '16', '32', '64', '128']
            data_size = [5, 10, 20, 40, 60, 80, 100, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_W = float(0.01)
            ptb = ['{}'.format(ptb_W)]
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t, data_size]]
            controls_W_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_W_e = make_control(control_name_e)
            controls = controls_W_t + controls_W_e
        elif data == 'KDDCUP99':
            test_mode_t = ['hst-t']
            test_mode_e = ['hst-e']
            n_t = ['1']
            n_e = ['1', '2', '4', '8', '16', '32', '64', '128']
            data_size = [5, 10, 20, 40, 60, 80, 100, 200, -1]
            data_size = [str(int(x)) for x in data_size]
            ptb = ['back', 'ipsweep', 'neptune', 'nmap', 'pod', 'portsweep', 'satan', 'smurf', 'teardrop',
                   'warezclient', 'unknown']
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t, data_size]]
            controls_W_t = make_control(control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_W_e = make_control(control_name_e)
            controls = controls_W_t + controls_W_e
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    mode = ['ptb', 'ds']
    data_name = ['MVN', 'RBM', 'EXP', 'KDDCUP99']
    # mode = ['ds']
    # data_name = ['KDDCUP99']
    controls = []
    for i in range(len(mode)):
        mode_i = mode[i]
        for j in range(len(data_name)):
            data_j = data_name[j]
            if mode_i in ['ptb'] and data_j in ['KDDCUP99']:
                continue
            if mode_i in ['ds'] and data_j in ['EXP']:
                continue
            if data_j in ['MVN', 'RBM', 'EXP']:
                model_j = data_j.lower()
            elif data_j in ['KDDCUP99']:
                model_j = 'rbm'
            else:
                raise ValueError('Not valid model')
            control_list = make_control_list(mode_i, data_j, model_j)
            controls = controls + control_list
    processed_result = process_result(controls)
    df_mean = make_df(processed_result, 'mean', False)
    df_history = make_df(processed_result, 'history', False)
    make_vis(df_history)
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
            base_result = load(base_result_path_i)['ht_state_dict']
            for metric_name in base_result:
                metric_name_ = 'test/{}'.format(metric_name)
                processed_result[metric_name_]['history'][exp_idx] = np.stack(base_result[metric_name], axis=0)
            for metric_name in ['test/fpr-threshold', 'test/fpr-error', 'test/fnr-threshold', 'test/fnr-error']:
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
        if metric_name in ['test/fpr-threshold', 'test/fpr-error', 'test/fnr-threshold', 'test/fnr-error']:
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


def make_vis(df_history):
    colors = plt.cm.get_cmap('tab10').colors
    n_e = ['1', '2', '4', '8', '16', '32', '64', '128']
    color_dict = {'lrt': {'t': {'1': colors[-1]},
                          'e': {n_e[i]: colors[i] for i in range(len(n_e))}},
                  'hst': {'t': {'1': colors[-1]},
                          'e': {n_e[i]: colors[i] for i in range(len(n_e))}}}
    linestyle_dict = {
        'lrt': {
            't': {'1': '-'},
            'e': {'1': '--',
                  '2': '-.',
                  '4': ':',
                  '8': (0, (5, 5)),
                  '16': (0, (5, 1)),
                  '32': (0, (1, 5)),
                  '64': (0, (3, 5, 5, 5)),
                  '128': (0, (3, 1, 1, 1, 1, 1))}
        },
        'hst': {
            't': {'1': '-'},
            'e': {'1': '--',
                  '2': '-.',
                  '4': ':',
                  '8': (0, (5, 5)),
                  '16': (0, (5, 1)),
                  '32': (0, (1, 5)),
                  '64': (0, (3, 5, 5, 5)),
                  '128': (0, (3, 1, 1, 1, 1, 1))}
        }
    }
    marker_dict = {'lrt-e': 'o', 'hst-e': 's', 'lrt-t': 'p', 'hst-t': 'd'}
    loc_dict = {'fpr': 'lower left', 'fnr': 'lower right'}
    fontsize_dict = {'legend': 10, 'label': 16, 'ticks': 16}
    metric_name_name_dict = {'fpr': 'Positive Error Exponent', 'fnr': 'Negative Error Exponent'}
    figsize = (5, 4)
    num_threshold = 3000
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        error_mode = metric_name.split('-')[0]
        mask = metric_name in ['fpr-error', 'fnr-error'] and stat == 'mean'
        if mask:
            ht_mode, n = df_name_list[2], df_name_list[4]
            ht_mode_list = df_name_list[2].split('-')
            df_name_threshold = '_'.join([*df_name_list[:-2], '{}-threshold'.format(error_mode), 'mean'])
            if len(df_name_list) == 7:
                fig_name = '_'.join([*df_name_list[:2], ht_mode_list[0], df_name_list[3], df_name_list[-2]])
            elif len(df_name_list) == 8:
                fig_name = '_'.join([*df_name_list[:2], ht_mode_list[0], df_name_list[3], df_name_list[5],
                                     df_name_list[-2]])
            else:
                raise ValueError('Not valid len')
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            x = df_history[df_name_threshold].iloc[0].to_numpy()
            x = x.reshape(-1, num_threshold)
            x = x.mean(axis=0)
            y = df_history[df_name].iloc[0].to_numpy()
            y = y.reshape(-1, num_threshold)
            if ht_mode_list[1] == 'e':
                n_ = float(df_name_list[4])
                y = 1 / n_ * np.log(y)
            else:
                y = np.log(y)
            y_mean = y.mean(axis=0)
            if len(y_mean[y_mean > -np.inf]) == 0:
                continue
            y_std = y.std(axis=0) / np.sqrt(y.shape[0])
            sorted_indices = np.argsort(x)
            x = x[sorted_indices]
            y_mean = y_mean[sorted_indices]
            y_std = y_std[sorted_indices]
            xlabel = 'Threshold'
            ylabel = metric_name_name_dict[error_mode]
            if ht_mode_list[1] == 't':
                label = 'Theoretical'
            else:
                label = 'n={}'.format(n)
            ax_1.plot(x, y_mean, label=label,
                      color=color_dict[ht_mode_list[0]][ht_mode_list[1]][n],
                      linestyle=linestyle_dict[ht_mode_list[0]][ht_mode_list[1]][n])
            ax_1.fill_between(x, (y_mean - y_std), (y_mean + y_std),
                              color=color_dict[ht_mode_list[0]][ht_mode_list[1]][n], alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
            ax_1.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
            if ht_mode_list[1] == 'e' and n == '128':
                ylim_mask = y_mean > -np.inf
                y_min = min(y_mean[ylim_mask])
                y_max = 0.01 * abs(y_min)
                ax_1.set_ylim([y_min, y_max])
                lines = ax_1.get_lines()
                x_min = []
                x_max = []
                for i in range(len(lines)):
                    x_value = lines[i].get_xdata()
                    y_value = lines[i].get_ydata()
                    xlim_mask_i = (y_value > y_min) & (y_value < y_max)
                    x_value = x_value[xlim_mask_i]
                    if len(x_value) > 0:
                        x_min.append(min(x_value))
                        x_max.append(max(x_value))
                if error_mode == 'fpr':
                    x_max = max(x_max)
                    x_min = max(x_min)
                    x_max = x_max + 0.1 * (x_max - x_min)
                elif error_mode == 'fnr':
                    x_max = min(x_max)
                    x_min = min(x_min)
                    x_min = x_min - 0.1 * (x_max - x_min)
                else:
                    raise ValueError('Not valid error mode')
                ax_1.set_xlim([x_min, x_max])
            ax_1.legend(fontsize=fontsize_dict['legend'], loc=loc_dict[error_mode])
    for fig_name in fig:
        fig_name_list = fig_name.split('_')
        data_name, ht_mode_0, metric_name = fig_name_list[0], fig_name_list[2], fig_name_list[-1]
        error_mode = metric_name.split('-')[0]
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        if len(fig_name_list) == 5:
            dir_name = 'ptb'
        elif len(fig_name_list) == 6:
            dir_name = 'ds'
        else:
            raise Value('Not valid len')
        dir_path = os.path.join(vis_path, dir_name, data_name, metric_name_name_dict[error_mode], ht_mode_0)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
