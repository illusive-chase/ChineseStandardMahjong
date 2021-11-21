# Chinese Standard Mahjong

### RL Framework: tianshou

+ learning
  + imitation_trainer.py: trainer, see imitate.py
  + imitation.py: policy of imitation learning wrapped for tianshou, see imitate.py
  + model.py: models of resnet
  + ppo.py: not used
  + wrapper.py: policy of neural network wrapped for env.runner.Runner, see imitate.py



### None-Learning Wrapper

+ env
  + bot.py: not used
  + imitator.py: wrapped environment for imitation learning, see imitate.py
  + runner.py: wrapped environment for Mahjong, see example.py



+ utils

  + distribution.py: for utils.policy.random_policy, see utils/policy.py
  + match_data.py: dump expert data for imitation learning, see preprocess.py
  + policy.py: implentation of wrapped policy
  + tile_traits.py: wrapper of player information to offer one-hot observation
  + vec_data.py: wrapper of player information to offer one-hot observation


