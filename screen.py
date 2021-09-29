import os

screen_name = 'csm'
expect_str = '*\\$ '
env_command = 'source activate ./venv'
python_script = 'python train.py'
arg_lists = {
    'lr': ['2.5e-5', '5e-5', '1e-4'],
    'hsize': [128, 256, 512],
    # 'vf-coef': [0.25, 0.5, 1.0, 2.0, 4.0],
    # 'ent-coef': ['1e-2', '1e-3'],
    # 'gamma': [0.95, 0.99, 0.995, 1.],
    # 'gae-lambda': [0.9, 0.95],
}
arg_fixed = {
    'log-dir': '0929',
}
args_collections = []

def combine_args(c, arg, exp_name, ks, vs, idx):
    if idx == len(ks):
        c.append(arg + ' --exp-name ' + exp_name)
        return
    for v in vs[idx]:
        combine_args(c, arg + ' --' + ks[idx] + ' ' + str(v), exp_name + '.' + str(v), ks, vs, idx + 1)

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
    f'expect "{expect_str}"',
    f'send "{env_command}\\n"',
    f'expect "{expect_str}"',
    'send "{}\\n"'.format('; '.join(assigned_command)),
    f'expect "{expect_str}"',
    'send "\\01"',
    'send "c"'
] for assigned_command in assigned_commands], [])[:-2] + [f'expect "{expect_str}"', 'send "\\01"', 'send "d"', f'expect "{expect_str}"', 'EOF', ''])

os.system(cmd)