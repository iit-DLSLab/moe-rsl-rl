<div style="text-align: left;">
  <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-green" alt="IsaacLab v2.3.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/rsl_rl%20-v3.3.0-brown" alt="rsl-rl v3.3.0" style="margin-bottom: 1px;">
</div>

# moe-rsl-rl
moe-rsl-rl is a reinforcement learning library that extends the Proximal Policy Optimization (PPO) implementation of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) to incorporate [Mixture-of-Expert](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf) (MoE).

Features:

- explicit expert selection
- explicit expert with different action size
- dense gating
- sparse gating
- top-k
- shared-layers between experts


## Installation

Install this package with:

```bash
pip install -e .
```

## How to use

See [here](https://github.com/iit-DLSLab/morphosymm-rl/blob/feature/main/README_how_to.md).

## Citing this work

If you find the work useful, please consider citing:

#### [Mixture-of-Experts RL for Fault-Tolerant Legged Locomotion (ArXiv)](https://arxiv.org/abs/2606.25965)

```
@inproceedings{turrisi2026moefault,
  author={Turrisi, Giulio and Pali, Ozan and Oneto, Luca and Semini, Claudio},
  booktitle={arXiv}, 
  title={Mixture-of-Experts RL for Fault-Tolerant Legged Locomotion}, 
  year={2026},
  doi={arXiv:2606.25965}
}
```

## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi)
