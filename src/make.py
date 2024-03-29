import argparse
import os
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--init_gpu', default=0, type=int)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiment', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--split_round', default=65535, type=int)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiment + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    init_gpu = args['init_gpu']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiment = args['num_experiment']
    resume_mode = args['resume_mode']
    mode = args['mode']
    data = args['data']
    model = args['model']
    split_round = args['split_round']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in
               list(range(init_gpu, init_gpu + num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiment, experiment_step))]
    world_size = [[world_size]]
    num_experiment = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}_{}'.format(run, mode, data)
    if mode == 'ptb':
        script_name = [['{}_ht.py'.format(run)]]
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
            controls_mean_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                            control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_mean_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                            control_name_e)
            ptb = []
            ptb_mean = float(0)
            ptb_logvar = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035,
                          0.04, 0.045, 0.05, 0.075, 0.1]
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_i = '{}-{}'.format(ptb_mean, ptb_logvar_i)
                ptb.append(ptb_i)
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t]]
            controls_logvar_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                              control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_logvar_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                              control_name_e)
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
            controls_W_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                         control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_W_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                         control_name_e)
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
            controls_tau_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                           control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e]]
            controls_tau_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                           control_name_e)
            controls = controls_tau_t + controls_tau_e
    elif mode == 'ds':
        script_name = [['{}_ht.py'.format(run)]]
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
            controls_mean_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                            control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_mean_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                            control_name_e)
            ptb_mean = float(0)
            ptb_logvar = float(0.01)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name_t = [[[data], [model], test_mode_t, ptb, n_t, data_size]]
            controls_logvar_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                              control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_logvar_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                              control_name_e)
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
            controls_W_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                         control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_W_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                         control_name_e)
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
            controls_W_t = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                         control_name_t)
            control_name_e = [[[data], [model], test_mode_e, ptb, n_e, data_size]]
            controls_W_e = make_controls(script_name, init_seeds, world_size, num_experiment, resume_mode,
                                         control_name_e)
            controls = controls_W_t + controls_W_e
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    s = '#!/bin/bash\n'
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiment {} ' \
                '--resume_mode {} --control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open(os.path.join('scripts', '{}_{}.sh'.format(filename, k)), 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        if not os.path.exists('scripts'):
            os.makedirs('scripts')
        run_file = open(os.path.join('scripts', '{}_{}.sh'.format(filename, k)), 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
