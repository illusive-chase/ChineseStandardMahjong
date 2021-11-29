# Chinese Standard Mahjong

## RL Framework: [tianshou](https://tianshou.readthedocs.io/zh/master/index.html)

+ learning
  + imitation_trainer.py: trainer, see test/test_imitator.py or imitate.py
  + imitation.py: policy of imitation learning wrapped for tianshou, see test/test_imitator.py or imitate.py
  + model.py: models of resnet
  + ppo.py: not used
  + wrapper.py: policy of neural network wrapped for env.runner.Runner, see imitate.py or scripts/botzone.py



## None-Learning Wrapper

+ env
  + bot.py: not used
  + imitator.py: wrapped environment for imitation learning, see imitate.py
  + multirunner.py: wrapped parallel environment for Mahjong, see test/test_multirunner.py or scripts/perf.py
  + runner.py: wrapped environment for Mahjong, see scripts/example.py


+ utils
  + distribution.py: for utils.policy.random_policy, see utils/policy.py
  + match_data.py: dump actions of expert data for imitation learning, see scripts/preprocess.py
  + merge.py: offer decorator to package the code files, see scripts/botzone.py
  + paired_data.py: dump observation-action pairs from match data for imitation learning, see scripts/generate_pair.py
  + policy.py: implentation of some wrapped policy
  + sync_buffer.py: a buffer for multi-thread control, see env/multirunner.py
  + tile_traits.py: wrapper of player information to offer one-hot observation
  + vec_data.py: wrapper of player information to offer one-hot observation


+ scripts
  + botzone.py: botzone interface
    + `python -m scripts.botzone -p data/best.pth -bn --merge`
  + example.py: examples of some API
    + `python -m scripts.example`
  + generate_pair.py: generate observation-action pairs from match data
    + `python -m scripts.generate_pair match.pkl -o pair.pkl  `
  + perf.py: evaluate performance of a certain model
    + `python -m scripts.perf -p A.pth -cp B.pth -bn`
  + preprocess.py: generate match data from match log (old version)
    + `python -m scripts.preprocess data.txt -o match.pkl`


+ test
  + test_augmentation.py: check whether the augmentation is correctly implemented
    + `python -m test.test_augmentation pair.pkl`
  + test_imitator.py: examples of imitation learning with match data
    + `python -m test.test_imitator -f match.pkl -d train -e experiment1 `
  + test_multirunner.py: check whether the multi-thread runner works correctly
    + `python -m test.test_multirunner` 
  + test_paired_data.py: check whether the encoder and decoder in utils/paired_data.py work correctly
    + `python -m test.test_paired_data match.pkl`
  

+ deprecated
  + some deprecated scripts (most of which are for performance reasons)



## Usage

### Train

```shell
python -m scripts.preprocess data.txt -o match.pkl
python -m scripts.generate_pair match.pkl -o pair.pkl
python imitate.py -f pair.pkl -d train -e experiment1
```

### Visualization

```shell
python imitate.py -f pair.pkl -d train -e experiment1 --eval
```

### Evaluate

```shell
python -m scripts.perf -p data/best.pth -cp data/cmp.pth -bn
```

### Upload

```shell
python -m scripts.botzone -p data/best.pth -bn --merge
```

+ Then upload the merge.zip as well as your best.pth onto the Botzone.