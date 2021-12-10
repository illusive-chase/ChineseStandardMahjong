from utils.paired_data import PairedDataset
from learning.wrapper import wrapper_policy
from env.multirunner import MultiRunner as MEnv
import argparse
import torch
from learning.model import *
import numpy as np
import tqdm
import os
import torch.multiprocessing as mp


def offline_pg(policy, other_policy, lr, dataset, batch_size, repeat, save_fn, force_to_save, best_reward):
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=lr)

    val_env = MEnv(policy, other_policy, 2, 1, 300, 200000, max_eps=600)
    policy.eval()
    other_policy.eval()

    batch, _ = dataset.sample(0)
    obs = batch['obs']
    mask = batch['mask']
    act = batch['gt_action']
    rew = batch['rew']
    base = rew.mean()
    logp_old = np.zeros((obs.shape[0],))
    indices = np.arange(obs.shape[0])
    np.random.shuffle(indices)

    with torch.no_grad():
        for idx in range(0, obs.shape[0], batch_size):
            indice = indices[idx:idx+batch_size]
            obs_tensor = torch.from_numpy(obs[indice]).to(policy.device).float()
            mask_tensor = torch.from_numpy(mask[indice]).to(policy.device)
            act_tensor = torch.from_numpy(act[indice]).to(policy.device)
            logits = policy.network(obs_tensor.view(obs_tensor.size(0), -1, 4, 9))
            logits = logits + (logits.min() - logits.max() - 20) * ~mask_tensor
            dist = torch.distributions.Categorical(logits=logits)
            logp_old[indice] = dist.log_prob(act_tensor).cpu().numpy()


    losses = []
    clips = []
    entropys = []
    
    for _ in range(repeat):
        for idx in range(0, obs.shape[0], batch_size):
            policy.eval()
            optimizer.zero_grad()
            indice = indices[idx:idx+batch_size]
            obs_tensor = torch.from_numpy(obs[indice]).to(policy.device).float()
            mask_tensor = torch.from_numpy(mask[indice]).to(policy.device)
            act_tensor = torch.from_numpy(act[indice]).to(policy.device)
            logp_old_tensor = torch.from_numpy(logp_old[indice]).to(policy.device)
            adv = torch.from_numpy(rew[indice]).to(policy.device) - base

            logits = policy.network(obs_tensor.view(obs_tensor.size(0), -1, 4, 9))
            logits = logits + (logits.min() - logits.max() - 20) * ~mask_tensor

            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(act_tensor) - logp_old_tensor).exp()
            surr1 = ratio * adv
            surr2 = ratio.clamp(0.8, 1.2) * adv
            clip_ratio = ((0.8 < ratio.detach()) & (ratio.detach() < 1.2)).float().mean()
            loss = -torch.min(surr1, surr2).mean()
            losses.append(loss.item())
            clips.append(clip_ratio.item())
            entropys.append(dist.entropy().mean().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.network.parameters(),
                max_norm=1,
                error_if_nonfinite=True
            )
            optimizer.step()
            policy.eval()
        np.random.shuffle(indices)

    print('loss/clip/entropy: {:.5f}/{:.3f}/{:.3f}'.format(np.mean(losses), np.mean(clips), np.mean(entropys)))

    val_env.collect(eval=True)
    reward = val_env.mean_reward()
    # print('update rew: {:.4f}'.format(reward))
    if force_to_save or reward > best_reward:
        save_fn(policy)
        return True, reward
    return False, reward




def train(args, best_reward):
    device = torch.device('cpu') if not args.cuda else torch.device(f'cuda:{args.cuda[0]}')
    dataset = PairedDataset(args.aug)
    fs = [args.file.replace('$', i) for i in '123456']
    fs = [open(f, 'rb') for f in fs]
    dataset.loads(fs)
    fs = [f.close() for f in fs]
    network = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    other = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)

    policy = wrapper_policy(network).to(device)
    other_policy = wrapper_policy(other).to(device)
    policy.load(args.path)
    other_policy.load(args.compare_path)
    other_policy.eval()

    return offline_pg(
        policy,
        other_policy,
        args.learning_rate,
        dataset,
        repeat=3,
        batch_size=args.batch_size,
        save_fn=lambda p: torch.save(p.state_dict(), args.dump),
        force_to_save=True,
        best_reward=best_reward
    )


def collect(args, device_idx):
    device = torch.device('cpu') if not device_idx else torch.device(f'cuda:{device_idx}')
    network = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    other = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    policy = wrapper_policy(network, deterministic=False).to(device)
    policy.load(args.path)
    policy.eval()

    other_policy = wrapper_policy(other).to(device)
    other_policy.load(args.compare_path)
    other_policy.eval()

    pdataset = PairedDataset()
    env = MEnv(policy, other_policy, 2, 1, 300, 200000, max_eps=2000)
    env.collect()
    buffer = env.get_buffer()
    batch, indices = buffer.sample(0)
    rew = 0
    for idx in range(batch['rew'].shape[0] - 1, -1, -1):
        if buffer.done[idx]:
            rew = int(batch['rew'][idx])
        pdataset.add(batch['obs'][idx].reshape(-1, 36), batch['act'][idx], batch['mask'][idx], rew)
    with open(args.file, 'wb') as f:
        pdataset.dump(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-p', '--path', type=str, default='')
    parser.add_argument('-d', '--dir', type=str, default='')
    parser.add_argument('-cu', '--cuda', type=str, default='')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2)
    parser.add_argument('-bs', '--batch-size', type=int, default=512)
    parser.add_argument('-dp', '--dropout', type=float, default=0)
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('-bn', '--batch-norm', action='store_true')
    parser.add_argument('--step', type=str, default='')
    parser.add_argument('--resnet', type=int, choices=[18, 34, 50, 101, 152], default=18)
    args = parser.parse_args()
    mp.set_start_method('spawn')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(3)
    
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if args.step:
        stage = int(args.step[0])
        step = int(args.step[1])
        curr = f'{stage}-{step}'
        next = f'{stage + step // 5}-{(step % 5) + 1}'
        args.path = os.path.join(args.dir, f'imp-{curr}.pth')
        args.compare_path = args.path
        args.file = os.path.join(args.dir, f'imp-{next}-$.pkl')
        args.dump = os.path.join(args.dir, f'imp-{next}.pth')
        accepted, rew = train(args, 0)
        print(f'[stage-{next}] rew: {rew:.4f}  {"--accepted" if accepted else "--rejected"}')

    else:
        for stage in range(1, 1000):
            args.compare_path = args.path
            best_reward = 0
            for mini_stage in range(1, 6):
                while True:
                    proc = []
                    for parallel in range(1, 7):
                        args.file = os.path.join(args.dir, f'imp-{stage}-{mini_stage}-{parallel}.pkl')
                        p = mp.Process(target=collect, args=(args, '' if not args.cuda else args.cuda[parallel]))
                        p.daemon = True
                        p.start()
                        proc.append(p)
                    for p in proc:
                        p.join()

                    args.file = os.path.join(args.dir, f'imp-{stage}-{mini_stage}-$.pkl')

                    args.dump = os.path.join(args.dir, f'imp-{stage}-{mini_stage}.pth')
                    accepted, rew = train(args, best_reward)
                    print(f'[stage-{stage}-{mini_stage}] rew: {rew:.4f}  {"--accepted" if accepted else "--rejected"}')
                    if accepted:
                        best_reward = rew
                        args.path = args.dump
                        break

