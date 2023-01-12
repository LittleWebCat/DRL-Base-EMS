# DRL-Base-EMS
DRL-Base-EMS for HEVs

<!-- GETTING STARTED -->
## Getting Started

The Vehicle Env is OpenAI-Gym like Env, so it needs to work with Gym Env files.

* You can simply install the base Gym library, use with the following command:
  ```sh
  $ pip install gym
  ```
  More information about Gym Env can find at [https://github.com/openai/gym](https://github.com/openai/gym)

The Algorithms library from Tianshou is currently hosted on PyPI and conda-forge. It requires Python >= 3.6.

* You can simply install Tianshou from PyPI with the following command:
  ```sh
  $ pip install tianshou
  ```
  More information about Tianshou library can find at [https://github.com/thu-ml/tianshou](https://github.com/thu-ml/tianshou)

### Setup

1. Get ready with Gym Env and Tianshou library
2. Download the `Vehicle_Env` file under `DRL-Base-EMS`
3. Open the Gym Env file of your workspace with suffix like `\gym\envs\`
4. Add all three content of `\DRL-Base-EMS\Vehicle_Env\classic_control\` to `\gym\envs\classic_control\`
5. Change the content of the `\gym\envs\classic_control\__init__.py` and `\gym\envs\__init__.py` by the instruction in `\DRL-Base-EMS\Vehicle_Env\__init__.py`
6. Now, you can try the different DRL Algorithms in your workspace.

## Acknowledgement
We acknowledge the following repositories that greatly shaped our implementation:
- https://github.com/thu-ml/tianshou for providing popular DRL Algorithms in PyTorch and operating guide. 
Please cite their work if you also find their code useful to your project:
```
@article{tianshou,
  title={Tianshou: A Highly Modularized Deep Reinforcement Learning Library},
  author={Weng, Jiayi and Chen, Huayu and Yan, Dong and You, Kaichao and Duburcq, Alexis and Zhang, Minghao and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2107.14171},
  year={2021}
}
```

## Citation
If you find our code useful to your work, please consider citing our paper:
```
@article{DRL-EMS,
  title={A comparative study of 13 deep reinforcement learning based energy management methods for a hybrid electric vehicle},
  author={Hanchen Wang, Bin Xu, Yiming Ye, Jiangfeng Zang},
  journal={https://www.sciencedirect.com/science/article/pii/S0360544222033837},
  year={2022}
}
```

## Contact
If you have any questions, please create an issue in this repository or contact Hanchen Wang (haw110@ou.edu)
