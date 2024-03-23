## Setup

See [installation.md](installation.md). It's worth going through this again since some dependencies have changed since homework 1. You also need to make sure to run `pip install -e .` in the hw2 folder.

## Running on Google Cloud
Starting with HW2, we will be providing some infrastructure to run experiments on Google Cloud compute. There are some very important caveats:

- **Do not leave your instance running.** The provided infrastructure tries to prevent this, but it will still be easy to accidentally leave your instance running and burn through all of your credits. You are responsible for making sure you use your credits wisely.
- **Only use this for big hyperparameter sweeps.** Definitely don't use Google Cloud for debugging; only launch a job once you are 100% sure your code works. Even then, single jobs will probably run faster on your local machine (yes, even if you don't have a GPU). The only reason to use Google Cloud is if you want to run multiple jobs in parallel.

For more instructions, see [google_cloud/README.md](google_cloud/README.md).

## Complete the code

There are TODOs in these files:

- `cs285/scripts/run_hw2.py`
- `cs285/agents/pg_agent.py`
- `cs285/networks/policies.py`
- `cs285/networks/critics.py`
- `cs285/infrastructure/utils.py`

See the [Assignment PDF](https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw2.pdf) for more info.

## Experiment commands

### Policy Gradients

```bash
# The small batch experiments
python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 1000 --exp_name cartpole --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 1000 -rtg --exp_name cartpole_rtg --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 1000 -na --exp_name cartpole_na --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na --video_log_freq -1

# The large batch experiments
python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 4000 --exp_name cartpole_lb --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 4000 -na --exp_name cartpole_lb_na --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name CartPole-v0 \
	-n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na --video_log_freq -1
```

### Using a Neural Network Baseline

```bash
# No baseline experiment
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
	-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah --video_log_freq -1

# Baseline experiment
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
	-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
	--use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --video_log_freq -1

# Run with a decreased number of bgs or blr
# baseline gradient steps
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
	-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
	--use_baseline -blr 0.01 -bgs 4 --exp_name cheetah_baseline --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
	-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
	--use_baseline -blr 0.01 -bgs 3 --exp_name cheetah_baseline --video_log_freq -1

# baseline learning rate
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
	-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
	--use_baseline -blr 0.02 -bgs 5 --exp_name cheetah_baseline --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
	-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
	--use_baseline -blr 0.03 -bgs 5 --exp_name cheetah_baseline --video_log_freq -1

# Add normalize advantages for better performance (and record video of HalfCheetah walking)
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
	-n 100 -b 5000 -na -rtg --discount 0.95 -lr 0.01 \
	--use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --video_log_freq 10
```

### Implement Generalized Advantage Estimation

```bash
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 \
	--ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
	--use_reward_to_go --use_baseline --gae_lambda 0 --exp_name lunar_lander_lambda0 --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name LunarLander-v2 \
	--ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
	--use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lunar_lander_lambda0.95 --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name LunarLander-v2 \
	--ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
	--use_reward_to_go --use_baseline --gae_lambda 0.98 --exp_name lunar_lander_lambda0.98 --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name LunarLander-v2 \
	--ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
	--use_reward_to_go --use_baseline --gae_lambda 0.99 --exp_name lunar_lander_lambda0.99 --video_log_freq -1

python cs285/scripts/run_hw2.py --env_name LunarLander-v2 \
	--ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
	--use_reward_to_go --use_baseline --gae_lambda 1 --exp_name lunar_lander_lambda1 --video_log_freq -1
```

### Hyperparameters and Sample Efficiency

```bash
for seed in $(seq 1 5); do
	python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
		--exp_name pendulum_default_s$seed \
		-rtg --use_baseline -na \
		--batch_size 5000 \
		--seed $seed --video_log_freq -1
done

# TODO
```

### Extra Credit: Humanoid

```bash
# TODO Run on Colab or Implement some optimizations
python cs285/scripts/run_hw2.py \
	--env_name Humanoid-v4 --ep_len 1000 \
	--discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001 \
	--baseline_gradient_steps 50 \
	-na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
	--exp_name humanoid --video_log_freq 5
```