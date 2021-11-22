from utils.merge import merger
import argparse

@merger
def botzone(args):
    import torch
    from learning.model import resnet18, resnet34, resnet50, resnet101, resnet152
    from learning.wrapper import wrapper_policy
    from env.bot import Bot
    torch.set_num_threads(1)
    device = torch.device('cpu')
    network = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm)
    policy = wrapper_policy(network).to(device)
    policy.load(args.pth)
    policy.eval()
    bot = Bot()
    bot.stepOneRound(policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pth', type=str, required=True)
    parser.add_argument('-bn', '--batch-norm', action='store_true')
    parser.add_argument('--resnet', type=int, choices=[18, 34, 50, 101, 152], default=18)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    botzone(args)