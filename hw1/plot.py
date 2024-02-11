import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to parse a single .tfevents file
def parse_tfevents_file(file_path):
    eval_average_return = []
    initial_data_collection_average_return = []
    eval_std_return = []
    for e in tf.compat.v1.train.summary_iterator(file_path):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_average_return.append(v.simple_value)
            elif v.tag == 'Initial_DataCollection_AverageReturn':
                initial_data_collection_average_return.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                eval_std_return.append(v.simple_value)
    return eval_average_return, initial_data_collection_average_return, eval_std_return

color_map = {
    'Ant-v4': 'red',
    'HalfCheetah-v4': 'blue',
    'Hopper-v4': 'green',
    'Walker2d-v4': 'purple',
}

# Your environments and corresponding directories
env_directories = {
    'Ant-v4': [
        # "q1_bc_ant_Ant-v4_12-02-2024_00-47-16",
        # "q1_bc_ant_Ant-v4_12-02-2024_01-06-25",
        # "q1_bc_ant_Ant-v4_12-02-2024_01-09-46",
        # "q1_bc_ant_Ant-v4_12-02-2024_02-57-03",
        # "q1_bc_ant_Ant-v4_12-02-2024_02-59-05",
        # "q1_bc_ant_Ant-v4_12-02-2024_03-14-48", # 200
        # "q1_bc_ant_Ant-v4_12-02-2024_03-15-07",
        # "q1_bc_ant_Ant-v4_12-02-2024_03-15-20",
        # "q1_bc_ant_Ant-v4_12-02-2024_03-15-32",
        "q2_dagger_ant_Ant-v4_12-02-2024_04-00-44",
    ],
    'HalfCheetah-v4': [
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_00-47-29",
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_01-06-45",
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_01-10-33",
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_02-57-34",
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_02-59-26",
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_03-16-15", # 200
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_03-16-26",
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_03-16-37",
        # "q1_bc_halfcheetah_HalfCheetah-v4_12-02-2024_03-16-47",
        "q2_dagger_halfcheetah_HalfCheetah-v4_12-02-2024_04-02-05",
    ],
    'Hopper-v4': [
        # "q1_bc_hopper_Hopper-v4_12-02-2024_00-47-46",
        # "q1_bc_hopper_Hopper-v4_12-02-2024_01-07-09",
        # "q1_bc_hopper_Hopper-v4_12-02-2024_01-10-50",
        # "q1_bc_hopper_Hopper-v4_12-02-2024_02-57-59",
        # "q1_bc_hopper_Hopper-v4_12-02-2024_02-59-48",
        # "q1_bc_hopper_Hopper-v4_12-02-2024_03-17-12", # 200
        # "q1_bc_hopper_Hopper-v4_12-02-2024_03-17-25",
        # "q1_bc_hopper_Hopper-v4_12-02-2024_03-17-34",
        # "q1_bc_hopper_Hopper-v4_12-02-2024_03-17-44",
        "q2_dagger_hopper_Hopper-v4_12-02-2024_04-04-20",
    ],
    'Walker2d-v4': [
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_00-47-57",
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_01-07-26",
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_01-11-05",
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_02-58-21",
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_03-00-10",
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_03-18-05", # 200
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_03-18-15",
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_03-18-25",
        # "q1_bc_walker2d_Walker2d-v4_12-02-2024_03-18-41",
        "q2_dagger_walker2d_Walker2d-v4_12-02-2024_04-05-11",
    ],
}

# Predefined "num agent train steps per iter" values
training_steps = [1000, 2000, 3000, 4000, 5000]
train_batch_size = [100, 200, 300, 400, 500]
iterations = list(range(1, 11))

# Base directory where .tfevents files are located
base_dir = './data'

data = {}

for env, dirs in env_directories.items():
    data[env] = {'Eval_AverageReturn': [], 'Initial_DataCollection_AverageReturn': [], 'Eval_StdReturn': []}
    for dir_name in dirs:
        full_dir_path = os.path.join(base_dir, dir_name)
        for root, dirs, files in os.walk(full_dir_path):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    file_path = os.path.join(root, file)
                    eval_average_return, initial_data_collection_average_return, eval_std_return = parse_tfevents_file(file_path)
                    if eval_average_return:
                        # data[env]['Eval_AverageReturn'].append(eval_average_return[-1])
                        data[env]['Eval_AverageReturn'].extend(eval_average_return)
                    if initial_data_collection_average_return:
                        # data[env]['Initial_DataCollection_AverageReturn'].append(initial_data_collection_average_return[-1])
                        data[env]['Initial_DataCollection_AverageReturn'].extend(initial_data_collection_average_return)
                    if eval_std_return:
                        # data[env]['Eval_StdReturn'].append(eval_std_return[-1])
                        data[env]['Eval_StdReturn'].extend(eval_std_return)

# Plotting
for env, metrics in data.items():
    eval_returns = metrics['Eval_AverageReturn']
    initial_returns = metrics['Initial_DataCollection_AverageReturn']
    eval_std_returns = metrics['Eval_StdReturn']
    env_color = color_map.get(env, 'black')
    if eval_returns and initial_returns and eval_std_returns:
        # plt.plot(training_steps, eval_returns, label=f'{env} Eval', color=env_color)
        # plt.hlines(initial_returns[0], training_steps[0], training_steps[-1], colors='k', linestyles='dashed', color=env_color)
        # plt.plot(train_batch_size, eval_returns, label=f'{env} Eval', color=env_color)
        plt.errorbar(iterations, eval_returns, yerr=eval_std_returns, label=f'{env} Eval', color=env_color)
        plt.hlines(initial_returns[0], iterations[0], iterations[-1], colors='k', linestyles='dashed')
        plt.xlabel('Iterations')
        plt.ylabel('Eval_AverageReturn')
        plt.title('DAgger')
        plt.legend(fontsize='large')
        plt.show()

# plt.xlabel('Num Agent Train Steps per Iter')
# plt.xlabel('Train batch size')
# plt.ylabel('Eval_AverageReturn')
# plt.title('Train batch size')
# plt.title('DAgger')
# plt.legend(fontsize='large')
# plt.show()
