import argparse
from env.runner import Runner as REnv
from utils.policy import inference_policy
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('match_file', type=str)
    parser.add_argument('-o', '--output', type=str, default='a.out')
    parser.add_argument('-i', '--match_id', type=str, default='')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    
    with open(args.output, 'w', encoding='utf-8') as fw:
        with open(args.match_file, 'r', encoding='utf-8') as f:
            line = f.readline()
            lines = []
            ith = 1
            while True:
                if not line or line[0:5] == 'Match':
                    if lines and (not args.match_id or args.match_id == lines[0].split()[1]):
                        assert lines[-1] == ''
                        assert lines[-2][:5] == 'Score'
                        lines.pop(-1)
                        policy = inference_policy(lines)
                        env = REnv(policy, args.verbose)
                        init_data = policy.as_match_data()
                        obs = env.reset(init_data)
                        done = False
                        policy.translate_fn = env.vec_data.translate
                        while not done:
                            env.render()
                            obs, rew, done, info = env.step(policy(obs))
                        env.render()
                        equal = bool((env.rew.astype(init_data.scores.dtype) == init_data.scores).all())
                        assert equal, init_data.match_id
                        policy.as_match_data().dump(fw)
                        print('{:05d}'.format(ith), init_data.match_id)
                        ith += 1
                    lines = []
                    if not line:
                        break
                lines.append(line.rstrip())
                line = f.readline()
