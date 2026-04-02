# MAPTD-Model-Agnostic-Predictive-Temperal-Difference
Learning to Predict and Control with Sparse Model Discovery and Deep Temporal Difference Reinforcement Learning

## What are we trying to do?
![WNO](/media/WNN_Neurips_INAE_Objective.png)

## NCWNO architecture in a glimpse.
![WNO](/media/ncwno.jpg)

## File description
```
📂 src
  |_📁 EQD            # Equation discovery files
  |_📁 MAPTD          # MAPTD algorithm with numerical predictor
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

