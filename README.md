## Approach 1: Curriculum Learning

NOTE: For consistent environment setups, use branch `curriculum`

Train - Curriculum Approach for all 3 Stages

```
python train.py
```

Train - Individual Curriculum Stages (This was used more frequently for development)

```
python train_easy2.py
python train_medium2.py
python train_hard2.py
```

Evaluation - Compare Reward for each Environment Between Models and Baselines

```
python evaluate2.py
```

Evaluation - Compare Networking Metrics with Baselines

```
python baseline.py
```

Evaluation - More In-depth & Reliable Metrics Comparison(Not Fully Completed)

```
python eval_metrics.py
```

## Approach 2: Agent Experimentation

To Run:

1. Update the environment config file
2. Train each model with `python train/train_{model_type}.py`
3. Plot the learning curves with `python plot_learning.py`
4. Run experiment with `python run_experiment.py`
