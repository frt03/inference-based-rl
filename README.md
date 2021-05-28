# Co-Adaptation of Algorithmic and Implementational Innovations in Inference-based Deep Reinforcement Learning

[[arxiv]](https://arxiv.org/abs/2103.17258)

This codebase includes inference-based off-policy algorithms, both KL control ([SAC](https://arxiv.org/abs/1801.01290)) and EM control ([MPO](https://arxiv.org/abs/1806.06920), [AWR](https://arxiv.org/abs/1910.00177), [AWAC](https://arxiv.org/abs/2006.09359)) methods.

If you use this codebase for your research, please cite the paper:
```
@article{furuta2021inference,
  title={Co-Adaptation of Algorithmic and Implementational Innovations in Inference-based Deep Reinforcement Learning},
  author={Hiroki Furuta and Tadashi Kozuno and Tatsuya Matsushima and Yutaka Matsuo and Shixiang Shane Gu},
  journal={arXiv preprint arXiv:arXiv:2103.17258},
  year={2021}
}
```

## Dependencies
We recommend you to use Docker. See [README](./docker/README.md) for setting up.

## Examples
See [examples](./examples) for the details.
```
python train_sac.py exp=HalfCheetah-v2 seed=0 gpu=0
python train_mpo.py exp=HalfCheetah-v2 seed=0 gpu=0
python train_awr.py exp=HalfCheetah-v2 seed=0 gpu=0
python train_awac.py exp=HalfCheetah-v2 seed=0 gpu=0
```

For ablation experiments (ELU or LayerNorm), use following command:
```
python train_sac2.py gpu=0 seed=0 env=Ant-v2 actor.nn_size=256 critic.nn_size=256 agent.architecture='nn2' agent.activation='elu' agent.use_layer_norm=False

python train_sac2.py gpu=0 seed=0 env=Ant-v2 actor.nn_size=256 critic.nn_size=256 agent.architecture='nn2' agent.activation='relu' agent.use_layer_norm=True

python train_sac2.py gpu=0 seed=0 env=Ant-v2 actor.nn_size=256 critic.nn_size=256 agent.architecture='nn2' agent.activation='elu' agent.use_layer_norm=True
```
For MPO w/o ELU and LayerNorm:
```
python train_mpo2.py gpu=0 seed=0 exp=Ant-v2
```

## Reference
This codebase is based on [PFRL](https://github.com/pfnet/pfrl).
