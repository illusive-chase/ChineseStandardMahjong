import os
import argparse

screen_name = 'csm'
env_command = 'source activate ./venv'
python_script = 'python train.py'
arg_lists = {
    # 'lr': ['1.25e-6', '2.5e-6', '5e-6', '1e-5'],
    # 'hsize': [128, 256, 512],
    'vf-coef': [0.5, 1.5, 4.5],
    # 'ent-coef': ['1e-2', '1e-3'],
    # 'gamma': [0.95, 0.99, 0.995, 1.],
    # 'gae-lambda': [0.9, 0.95],
    # 'seed': [1, 2],
}
arg_fixed = {
    'log-dir': '1003',
    'max-epoch': 1000,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['boot', 'shutdown'])
    args = parser.parse_args()
    if args.mode == 'boot':
        presence = int(eval(os.popen('screen -ls 2> /dev/null | grep csm | wc -l').read()))
        if presence:
            raise RuntimeError("ALREADY PRESENT")

        
        prompt = '*\\$ '
        args_collections = []
        def combine_args(c, arg, exp_name, ks, vs, idx):
            if idx == len(ks):
                c.append(arg + ' --exp-name ' + exp_name)
                return
            for v in vs[idx]:
                combine_args(c, arg + ' --' + ks[idx] + ' ' + str(v), exp_name + '.' + ks[idx] + '.' + str(v), ks, vs, idx + 1)

        combine_args(args_collections, python_script + ''.join([' --' + k + ' ' + str(v) for k, v in arg_fixed.items()]), 'exp', list(arg_lists.keys()), list(arg_lists.values()), 0)

        with open('screen.txt', 'w') as f:
            f.write('\n'.join(args_collections))

        assigned_commands = []
        for i in range(10):
            if args_collections == []:
                break
            if len(assigned_commands) <= i:
                assigned_commands.append([])
            assigned_commands[i].append(args_collections.pop(0))

        cmd = '\n'.join([
            'expect <<EOF',
            'set timeout 3',
            f'spawn screen -S {screen_name}',
        ] + sum([[
            f'expect "{prompt}"',
            f'send "{env_command}\\n"',
            f'expect "{prompt}"',
            'send "{}\\n"'.format('; '.join(assigned_command)),
            f'expect "{prompt}"',
            'send "\\01"',
            'send "c"'
        ] for assigned_command in assigned_commands], [])[:-2] + [f'expect "{prompt}"', 'send "\\01"', 'send "d"', f'expect "{prompt}"', 'EOF', ''])
        
        os.system(cmd)
    else:
        cmd = '\n'.join([
            'expect <<EOF',
            'set timeout 3',
            f'spawn screen -r {screen_name}',
            'send "\\01"',
            'send ":"',
            'expect eof',
            'send "quit"',
            'EOF',
            ''
        ])
        
        os.system(cmd)