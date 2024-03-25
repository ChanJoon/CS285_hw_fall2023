import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline


def find_directories_with_word(path, word):
    """Returns a list of directories within 'path' that include 'word'."""
    p = Path(path)
    # Using a case-insensitive search for 'word' in directory names
    directories_with_word = [d.name for d in p.iterdir() if d.is_dir() and word.lower() in d.name.lower()]
    return directories_with_word

# Function to parse a single .tfevents file
def parse_tfevents_file(file_path):
    total_steps = []
    eval_average_return = []
    initial_data_collection_average_return = []
    eval_std_return = []
    baseline_loss = []
    for e in tf.compat.v1.train.summary_iterator(file_path):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_average_return.append(v.simple_value)
                total_steps.append(e.step)
            elif v.tag == 'Initial_DataCollection_AverageReturn':
                initial_data_collection_average_return.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                eval_std_return.append(v.simple_value)
            elif v.tag == 'Baseline_Loss':
                baseline_loss.append(v.simple_value)
    return total_steps, eval_average_return, initial_data_collection_average_return, eval_std_return, baseline_loss

color_map = {
    'Ant-v4': 'red',
    'HalfCheetah-v4': 'blue',
    'Hopper-v4': 'green',
    'Walker2d-v4': 'purple',
}
data = {}

def plot(base_dir, env_prefix, x_axis, y_axis, title, exp_name_len):
    for i in range(len(env_prefix)):
        env = env_prefix[i]
        dirs = find_directories_with_word(base_dir, env)
        print(f"Find directories with word '{env}': {dirs}")
        data[env] = {'Eval_AverageReturn': [], 'Initial_DataCollection_AverageReturn': [], 'Eval_StdReturn': [], 'Baseline_Loss': []}
        for dir_name in dirs:
            full_dir_path = os.path.join(base_dir, dir_name)
            for root, dirs, files in os.walk(full_dir_path):
                for file in files:
                    if file.startswith('events.out.tfevents'):
                        file_path = os.path.join(root, file)
                        total_steps, eval_average_return, initial_data_collection_average_return, eval_std_return, baseline_loss = parse_tfevents_file(file_path)
                        if eval_average_return:
                            data[env]['Eval_AverageReturn'].extend(eval_average_return)
                        if initial_data_collection_average_return:
                            data[env]['Initial_DataCollection_AverageReturn'].extend(initial_data_collection_average_return)
                        if eval_std_return:
                            data[env]['Eval_StdReturn'].extend(eval_std_return)
                        if baseline_loss:
                            data[env]['Baseline_Loss'].extend(baseline_loss)

    # Plotting
    for env, metrics in data.items():
        eval_returns = metrics['Eval_AverageReturn']
        initial_returns = metrics['Initial_DataCollection_AverageReturn']
        eval_std_returns = metrics['Eval_StdReturn']
        baseline_loss = metrics['Baseline_Loss']
        env_color = color_map.get(env, 'black')
        if eval_returns and initial_returns and eval_std_returns:
            # x_smooth = np.linspace(total_steps[0], total_steps[-1], 30)
            # spl = make_interp_spline(total_steps, eval_returns, k=3)
            # y_smooth = spl(x_smooth)
            # plt.plot(x_smooth, y_smooth, label=f'{env[exp_name_len:]} Eval')
            # plt.text(total_steps[-1], eval_returns[-1], f'{eval_returns[-1]:.2f}', fontsize=12)
            # plt.plot(total_steps, eval_returns, label=f'{env[exp_name_len:]}')
            # plt.hlines(initial_returns[0], training_steps[0], training_steps[-1], colors='k', linestyles='dashed', color=env_color)
            # plt.hlines(0, total_steps[0], total_steps[-1], colors='k', linestyles='solid', color=env_color)
            # plt.text(0.1, 0.5, '0')
            # plt.hlines(300, total_steps[0], total_steps[-1], colors='k', linestyles='dashed', color=env_color)
            # plt.text(0.1, 300.5, '300')
            plt.fill_between(total_steps, 
                            [a - b for a, b in zip(eval_returns, eval_std_returns)], 
                            [a + b for a, b in zip(eval_returns, eval_std_returns)], 
                            alpha=0.2)
            plt.text(total_steps[-1], eval_returns[-1], f'{eval_returns[-1]:.2f}', fontsize=12)

    # Fullscreen 1920x1080
    plt.gcf().set_size_inches(19.2, 10.8) 
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Base directory where .tfevents files are located
    base_dir = './data'

    env_prefix_pg_small_batch = [
        'cartpole_CartPole-v0',
        'cartpole_rtg_CartPole-v0',
        'cartpole_na_CartPole-v0',
        'cartpole_rtg_na_CartPole-v0'
    ]

    env_prefix_pg_large_batch = [
        'cartpole_lb_CartPole-v0',
        'cartpole_lb_rtg_CartPole-v0',
        'cartpole_lb_na_CartPole-v0',
        'cartpole_lb_rtg_na_CartPole-v0'
    ]
    
    env_prefix_baseline_loss = ['cheetah_baseline_HalfCheetah-v4']
    
    env_prefix_cheetah = [
        'cheetah_HalfCheetah-v4',
        'cheetah_baseline_HalfCheetah-v4',
    ]
    
    env_prefix_bgs_halfcheetah = [
        'baseline_bgs3_HalfCheetah-v4',
        'baseline_bgs4_HalfCheetah-v4',
        'baseline_HalfCheetah-v4',
    ]
    env_prefix_blr_halfcheetah = [
        'baseline_HalfCheetah-v4',
        'baseline_blr2_HalfCheetah-v4',
        'baseline_blr3_HalfCheetah-v4',
    ]
    
    env_prefix_lambda = [
        'lambda0_LunarLander-v2',
        'lambda0.95_LunarLander-v2',
        'lambda0.98_LunarLander-v2',
        'lambda0.99_LunarLander-v2',
        'lambda1_LunarLander-v2',
    ]
    
    env_prefix_inverted_pendulum = [
        'default_s1_InvertedPendulum-v4',
        'default_s2_InvertedPendulum-v4',
        'default_s3_InvertedPendulum-v4',
        'default_s4_InvertedPendulum-v4',
        'default_s5_InvertedPendulum-v4',
    ]
    
    exp_name_len = len('default_')
    
    ## HW2 5. Learning curve for the baseline loss
    # plot(base_dir, env_prefix_baseline_loss, x_axis='Environment steps', y_axis='Baseline_Loss', title='Baseline Loss in HalfCheetah-v4', exp_name_len=exp_name_len)
    
    ## HW2 5. Learning curve for the eval return
    plot(base_dir, env_prefix=env_prefix_lambda, x_axis='Environment steps', y_axis='Eval_AverageReturn', title='Avg. Return in LunaLander-v2 for different lambdas', exp_name_len=exp_name_len)