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
from copy import deepcopy
import random

def offline_ppo(args, critic, policy, other_policy, target_policy, dataset, batch_size, repeat, seed_range):
    
    try:
        optimizer = torch.optim.Adam()
        sd = torch.load(args.path.replace('.pth', '.opt.pth'), map_location='cpu')
        optimizer.load_state_dict(sd)
        print('opt load @', args.path.replace('.pth', '.opt.pth'))
    except:
        optimizer = torch.optim.Adam(policy.network.parameters(), lr=args.learning_rate)

    critic_optimizer = torch.optim.Adam(critic.network.parameters(), lr=args.critic_learning_rate)

    val_env = MEnv(policy, other_policy, 2, 1, 300, 200000, max_eps=100)
    policy.eval()
    other_policy.eval()

    batch, _ = dataset.sample(0, shuffle=False)
    assert dataset.augmentation == 1
    obs = batch['obs']
    mask = batch['mask']
    act = batch['gt_action']
    rew = batch['rew'] * 0.1
    '''rew = np.zeros(batch['rew'].shape)
    rew[batch['rew'] < -8] = -2
    rew[batch['rew'] == -8] = -1
    rew[batch['rew'] == 0] = -1
    rew[batch['rew'] > 0] = (0.5 * batch['rew'][batch['rew'] > 0]) ** 0.5'''
    done = np.zeros(rew.shape, dtype=np.bool)
    tile_count = obs[:, 0:4].reshape(-1, 4 * 36).sum(1)
    done[:-1] = tile_count[1:] < tile_count[:-1]
    done[-1] = True
    print('episode', done.sum())
    del tile_count

    rew[~done] = 0

    obs_next = np.empty_like(obs)
    obs_next[:-1] = obs[1:]

    logp_old = np.zeros((obs.shape[0],))
    vs = np.zeros((obs.shape[0],))
    vs_ = np.zeros((obs.shape[0],))
    adv = np.zeros((obs.shape[0],))

    indices = np.arange(obs.shape[0])
    np.random.shuffle(indices)

    with torch.no_grad():
        for idx in range(0, obs.shape[0], batch_size):
            indice = indices[idx:idx+batch_size]
            obs_tensor = torch.from_numpy(obs[indice]).to(policy.device).float()
            obs_tensor = obs_tensor.view(obs_tensor.size(0), -1, 4, 9)
            mask_tensor = torch.from_numpy(mask[indice]).to(policy.device)
            act_tensor = torch.from_numpy(act[indice]).to(policy.device)
            done_tensor = torch.from_numpy(done[indice]).to(policy.device)
            rew_tensor = torch.from_numpy(rew[indice]).to(policy.device).float()
            logits = policy.network(obs_tensor.view(obs_tensor.size(0), -1, 4, 9))
            logits = logits + (logits.min() - logits.max() - 40) * ~mask_tensor
            dist = torch.distributions.Categorical(logits=logits)
            logp_old[indice] = dist.log_prob(act_tensor).cpu().numpy()
    
    del done_tensor
    del rew_tensor


    losses = []
    clips = []
    entropys = []
    
    for rpt in range(repeat):

        only_critic = rpt + 5 < repeat

        with torch.no_grad():
            critic.eval()
            for idx in range(0, obs.shape[0], batch_size):
                indice = indices[idx:idx+batch_size]
                obs_tensor = torch.from_numpy(obs[indice]).to(policy.device).float()
                obs_tensor = obs_tensor.view(obs_tensor.size(0), -1, 4, 9)
                obs_next_tensor = torch.from_numpy(obs_next[indice]).to(policy.device).float()
                obs_next_tensor = obs_tensor.view(obs_next_tensor.size(0), -1, 4, 9)
                vs[indice] = critic.network(obs_tensor).flatten().cpu().numpy()
                vs_[indice] = critic.network(obs_next_tensor).flatten().cpu().numpy()
            
            gamma = 1
            gae_lambda = 0.9
            delta = rew + vs_ * gamma - vs
            m = (1.0 - done.astype(np.float64)) * (gamma * gae_lambda)
            gae = 0.0
            for i in range(len(adv) - 1, -1, -1):
                gae = delta[i] + m[i] * gae
                adv[i] = gae
            returns = adv + vs

        del obs_next_tensor


        for idx in range(0, obs.shape[0], batch_size):
            policy.eval()
            if only_critic:
                critic.train()
                critic_optimizer.zero_grad()
            else:
                critic.eval()
                optimizer.zero_grad()
            indice = indices[idx:idx+batch_size]
            obs_tensor = torch.from_numpy(obs[indice]).to(policy.device).float()
            obs_tensor = obs_tensor.view(obs_tensor.size(0), -1, 4, 9)
            mask_tensor = torch.from_numpy(mask[indice]).to(policy.device)
            act_tensor = torch.from_numpy(act[indice]).to(policy.device)
            logp_old_tensor = torch.from_numpy(logp_old[indice]).to(policy.device)
            adv_tensor = torch.from_numpy(adv[indice]).to(policy.device)
            returns_tensor = torch.from_numpy(returns[indice]).to(policy.device)

            

            if only_critic:
                value = critic.network(obs_tensor).flatten()
                vf_loss = (returns_tensor - value).pow(2).mean()
                loss = vf_loss
                # print('vf: {:.3f}'.format(vf_loss.item()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.network.parameters(),
                    max_norm=0.5,
                    error_if_nonfinite=True
                )
                critic_optimizer.step()
            else:
                logits = policy.network(obs_tensor)
                logits = logits + (logits.min() - logits.max() - 40) * ~mask_tensor

                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(act_tensor) - logp_old_tensor).exp()
                surr1 = ratio * adv_tensor
                surr2 = ratio.clamp(0.8, 1.2) * adv_tensor
                clip_ratio = ((0.8 < ratio.detach()) & (ratio.detach() < 1.2)).float().mean()
                clip_loss = -torch.min(surr1, surr2).mean()
                ent_loss = dist.entropy().mean()
                loss = clip_loss - 1e-5 * ent_loss
                losses.append(loss.item())
                clips.append(clip_ratio.item())
                entropys.append(ent_loss.item())
                # print('clip/entropy/cr: {:.5f}/{:.3f}/{:.3f}'.format(clip_loss.item(), ent_loss.item(), clip_ratio.item()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.network.parameters(),
                    max_norm=0.5,
                    error_if_nonfinite=True
                )
                optimizer.step()
            policy.eval()
        np.random.shuffle(indices)

    print('mean clip/entropy/cr: {:.5f}/{:.3f}/{:.3f}'.format(np.mean(losses), np.mean(entropys), np.mean(clips)))
    val_env.collect(eval=True, varity=seed_range)
    reward = val_env.mean_reward()
    print('val rew: {:.4f}'.format(reward))
    if reward > 0:
        for i in range(5):
            for idx in range(0, obs.shape[0], batch_size):
                target_policy.eval()
                indice = indices[idx:idx+batch_size]
                obs_tensor = torch.from_numpy(obs[indice]).to(target_policy.device).float()
                obs_tensor = obs_tensor.view(obs_tensor.size(0), -1, 4, 9)
                mask_tensor = torch.from_numpy(mask[indice]).to(target_policy.device)
                act_tensor = torch.from_numpy(act[indice]).to(target_policy.device)
                logp_old_tensor = torch.from_numpy(logp_old[indice]).to(target_policy.device)
                adv_tensor = torch.from_numpy(adv[indice]).to(target_policy.device)

                logits = target_policy.network(obs_tensor)
                logits = logits + (logits.min() - logits.max() - 40) * ~mask_tensor

                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(act_tensor) - logp_old_tensor).exp()
                surr1 = ratio * adv_tensor
                surr2 = ratio.clamp(0.8, 1.2) * adv_tensor
                clip_ratio = ((0.8 < ratio.detach()) & (ratio.detach() < 1.2)).float().mean()
                clip_loss = -torch.min(surr1, surr2).mean()
                ent_loss = dist.entropy().mean()
                loss = (clip_loss - 1e-5 * ent_loss) * 2
                loss.backward()
            np.random.shuffle(indices)

    return reward

def train(args, target_policy, seed_range):
    device = torch.device('cpu') if not args.cuda else torch.device(f'cuda:{args.cuda[0]}')
    dataset = PairedDataset(args.aug)
    fs = [args.file.replace('$', str(i)) for i in range(1, len(args.cuda))]
    fs = [open(f, 'rb') for f in fs]
    dataset.loads(fs)
    fs = [f.close() for f in fs]
    
    policy = deepcopy(target_policy).to(device)
    other = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    other_policy = wrapper_policy(other).to(device)
    other_policy.load(args.compare_path)
    other_policy.eval()
    print('opponent load @', args.compare_path)

    critic = wrapper_policy(resnet18(use_bn=True, dropout=0., shape=(145, 1))).to(device)

    return offline_ppo(
        args,
        critic,
        policy,
        other_policy,
        target_policy,
        dataset,
        repeat=10,
        batch_size=args.batch_size,
        seed_range=seed_range
    )


def collect(args, device_idx, seed_range):
    device = torch.device('cpu') if not device_idx else torch.device(f'cuda:{device_idx}')
    network = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    other = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    policy = wrapper_policy(network, deterministic=False).to(device)
    policy.load(args.path)
    policy.eval()

    other_policy = wrapper_policy(other).to(device)
    other_policy.load(args.compare_path)
    other_policy.eval()

    print('policy load @', args.path)
    print('opponent load @', args.compare_path)

    pdataset = PairedDataset()
    env = MEnv(policy, other_policy, 2, 1, 300, 200000, max_eps=1000)
    env.collect(varity=seed_range)
    buffer = env.get_buffer()
    batch, indices = buffer.sample(0)
    rew = np.copy(batch['rew']).astype(np.uint8)
    last_rew = 0
    for idx in range(rew.shape[0] - 1, -1, -1):
        if buffer.done[idx]:
            last_rew = rew[idx]
        else:
            rew[idx] = last_rew
    for idx in range(rew.shape[0]):
        pdataset.add(batch['obs'][idx].reshape(-1, 36), batch['act'][idx], batch['mask'][idx], rew[idx])
    with open(args.file, 'wb') as f:
        pdataset.dump(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-va', '--varity', type=int, default=10000)
    parser.add_argument('-p', '--path', type=str, default='')
    parser.add_argument('-d', '--dir', type=str, default='')
    parser.add_argument('-cu', '--cuda', type=str, default='')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-clr', '--critic-learning-rate', type=float, default=1e-3)
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

    mini_stage_num = 10

    if args.step:
        stage = int(args.step)
        curr = f'{stage-1}-{mini_stage_num}'
        next = f'{stage}-{mini_stage_num}'
        if stage != 1:
            args.path = os.path.join(args.dir, f'imp-{curr}.pth')
        args.compare_path = args.path
        args.file = os.path.join(args.dir, f'imp-{next}-$.pkl')
        args.dump = os.path.join(args.dir, f'imp-{next}.pth')
        device = torch.device('cpu') if not args.cuda else torch.device(f'cuda:{args.cuda[0]}')
        target_policy = wrapper_policy(eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)).to(device)
        target_policy.load(args.path)
        rew = train(args, target_policy, (0, args.varity))
        print(f'[stage-{next}] rew: {rew:.4f}')

    else:
        rand = random.Random(args.seed)
        args.compare_path = args.path
        args.last_path = args.path
        best_reward = 0.01
        for stage in range(1, 1000):
            while True:
                device = torch.device('cpu') if not args.cuda else torch.device(f'cuda:{args.cuda[0]}')
                target_policy = wrapper_policy(eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)).to(device)
                target_policy.load(args.path)
                other = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
                other_policy = wrapper_policy(other).to(device)
                other_policy.load(args.compare_path)

                try:
                    optimizer = torch.optim.Adam()
                    sd = torch.load(args.path.replace('.pth', '.opt.pth'), map_location='cpu')
                    optimizer.load_state_dict(sd)
                    print('opt load @', args.path.replace('.pth', '.opt.pth'))
                except:
                    optimizer = torch.optim.Adam(target_policy.network.parameters(), lr=args.learning_rate)


                optimizer.zero_grad()

                for mini_stage in range(1, mini_stage_num + 1):

                    args.file = os.path.join(args.dir, f'imp-{stage}-{mini_stage}-$.pkl')
                    args.dump = os.path.join(args.dir, f'imp-{stage}-{mini_stage}.pth')
                    base = rand.randint(0, 1000) * args.varity
                    seed_range = (base, base + args.varity)

                    proc = []
                    for parallel, cuda_idx in enumerate(args.cuda[1:]):
                        args.file = os.path.join(args.dir, f'imp-{stage}-{mini_stage}-{parallel + 1}.pkl')
                        p = mp.Process(target=collect, args=(args, cuda_idx, seed_range))
                        p.daemon = True
                        p.start()
                        proc.append(p)
                    for p in proc:
                        p.join()

                    train(args, target_policy, seed_range)
                        

                optimizer.step()


                # eval

                eval_env = MEnv(target_policy, other_policy, 2, 1, 300, 200000, max_eps=3000)
                target_policy.eval()
                other_policy.eval()
                eval_env.collect(eval=True)
                rew = eval_env.mean_reward()
                accepted = rew > best_reward
                print(f'[stage-{stage}] rew: {rew:.4f}  {"--accepted" if accepted else "--rejected"}')
                if accepted:
                    best_reward = rew
                    torch.save(target_policy.state_dict(), args.dump)
                    torch.save(optimizer.state_dict(), args.dump.replace('.pth', '.opt.pth'))
                    args.path = args.dump
                    args.last_path = args.path
                    break
                args.path = args.last_path

