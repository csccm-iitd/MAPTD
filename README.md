# MAPTD-Model-Agnostic-Predictive-Temperal-Difference
Learning to Predict and Control with Sparse Model Discovery and Deep Temporal Difference Reinforcement Learning

## Architecture
![MAPTD](/media/MAPTD.jpg)

## MAPTD agent taking action to actively control vibration of mechanical systems
![Beam](/media/Beam_control_2d.gif)
![Structure](/media/STR_control_2d.gif)

## Inference time comparison between model-based control algorithms
![Beam](/media/beam_inference_time.png)
![Structure](/media/STR_inference_time.png)

## File description
```
📂 src
  |_📁 EQD            # Equation discovery files
  |_📂 MAPTD          # MAPTD algorithm with numerical predictor
    |_📁 algorithm             # contains main files for reinforcement learning
    |_📁 cfgs                  # contains yaml files to configure the test examples
    |_📁 data                  # contains model information of 76DOF structure
    |_📁 logs                  # contains trained agents and logs
    |_📁 results               # directory to save trajectories
    |_📄 cfg.py                # file to purge yaml files
    |_📄 dynamics_gym.py       # script for accumulating all the environments
    |_📄 env_beam.py           # cantilever beam environment setup
    |_📄 envs.py               # environment wrapper
    |_📄 env_tallstorey.py     # 76DOF environment setup
    |_📄 logger.py             # file to log training information
    |_📄 train_maptd_76dof.py  # training script
    |_📄 train_maptd_beam.py   # training script
    |_📄 test_76dof.py         # testing script
    |_📄 test_beam.py          # testing script
  |_📁 MAPTD_hybrid   # MAPTD algorithm with hybrid Real2Sim strategy
  |_📁 MAPTD_NN       # MAPTD algorithm with ANN as world model
  |_📁 MAPTD_oml      # MAPTD using hybrid Real2Sim strategy with online NO update
  |_📁 MPC            # Model Predictive Control algorithm true physics
  |_📁 NO             # Neural Operator surrogate
  |_📁 TDMPC          # TDMPC algorithm 
  |_📂 data 
    |_📄 B76_inp.mat  # 76DOF benchmark model
  |_📂 images         # Result directory
  |_📄 beam_solver.py         # script of finite element method
  |_📄 piezoelectric.py       # script for estimating the voltage supply in the piezoelectric patch
  |_📄 Systems_76dof.py       # script for 76 DOF structure
  |_📄 Systems_76dof_rom.py   # script for ROM of 76 DOF structure
  |_📄 Systems_cantilever.py  # script for cantilever beam
  |_📄 utils_data.py          # script for generating data
  |_📄 wind_pressure.py       # script for generating wind pressure
|_📄 TT_rlc.yml       # Anaconda environment configuration details
```

## Essential Python Libraries
  + Install the library dependencies from the `TT_rlc.yml` file

