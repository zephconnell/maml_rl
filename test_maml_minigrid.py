from rllab.envs.mini_grid_env_rand import MiniGridEnvRand
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.envs.base import TfEnv
from load_condensed_genomes import load_condensed_genomes

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf

stub(globals())

# horizon of 10
initial_params_file1 = 'data/local/trpo-maml-minigrid/trpo_maml1_fbs20_mbs60_flr0.1_metalr0.01_step1/params.pkl'

test_num_mazes = 1
genomes_file = 'attempt6_envs_condensed.txt'
configs = load_condensed_genomes(genomes_file)
if len(configs) < test_num_mazes:
    test_num_mazes = len(configs)
num_mazes = 1

# ICML values
step_sizes = [0.1]
initial_params_files = [initial_params_file1]
gen_name = 'icml_minigrid_results_'
names = ['maml']

exp_names = [gen_name + name for name in names]

all_avg_returns = []
for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
    avg_returns = []
    config = [None]*1
    for i in range(test_num_mazes):
        config[0] = configs[i]
        env = TfEnv(normalize(MiniGridEnvRand(configs=config,num_mazes=num_mazes)))
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = VPG(
            env=env,
            policy=None,
            load_policy=initial_params_file,
            baseline=baseline,
            batch_size=100,
            max_path_length=10,
            n_itr=2,
            optimizer_args={'init_learning_rate': step_sizes[step_i], 'tf_optimizer_args': {'learning_rate': 0.5*step_sizes[step_i]}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
        )

        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=1,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            exp_prefix='trpo_minigrid_test',
            exp_name='test',
            #plot=True,
        )
        import pdb; pdb.set_trace()
        # get return from the experiment
        with open('data/local/trpo-minigrid-test/test/progress.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            row = None
            returns = []
            for row in reader:
                i+=1
                if i ==1:
                    assert row[-1] == 'AverageReturn'
                else:
                    returns.append(float(row[-1]))
            avg_returns.append(returns)
    all_avg_returns.append(avg_returns)


for i in range(len(initial_params_files)):
    returns = []
    std_returns = []
    task_avg_returns = []
    for itr in range(len(all_avg_returns[i][0])):
        returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
        std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))

        task_avg_returns.append([ret[itr] for ret in all_avg_returns[i]])

    results = {'task_avg_returns': task_avg_returns}
    with open(exp_names[i] + '.pkl', 'w') as f:
        pickle.dump(results, f)

import pdb; pdb.set_trace()

