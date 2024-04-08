from collections import deque
from functools import partial
from itertools import chain

import config
import doy
import env_utils
import paths
import ppo
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from doy import PiecewiseLinearSchedule as PLS
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from utils import create_decoder

state_dict = torch.load(paths.get_latent_policy_path(config.get().exp_name))
cfg = config.get(base_cfg=state_dict["cfg"], reload_keys=["stage3"])
cfg.stage_exp_name += doy.random_proquint(1)
doy.print("[bold green]Running LAPO stage 3 (latent policy decoding) with config:")
config.print_cfg(cfg)

policy = utils.create_policy(
    cfg.model,
    action_dim=cfg.model.la_dim,
    state_dict=state_dict["policy"],
    strict_loading=True,
)

policy.decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=(192, 128, 64),
)

# we only need the IDM for training a decoder online
# if we just do RL, we don't need it
models_path = paths.get_models_path(cfg.exp_name)
idm, _ = utils.create_dynamics_models(cfg.model, state_dicts=torch.load(models_path))
idm.eval()

## ---- hacky stuff to monkey patch policy arch ----
policy.policy_head_sl = policy.policy_head
policy.policy_head_rl = nn.Linear(
    policy.policy_head_sl.in_features, cfg.model.ta_dim
).to(config.DEVICE)
# init rl path to 0 output
with torch.no_grad():
    policy.policy_head_rl.weight[:] = 0
    policy.policy_head_rl.bias[:] = 0

policy.fc_rl = policy.fc
policy.fc_sl = nn.Sequential(
    nn.Linear(policy.fc.in_features, policy.fc.out_features), nn.ReLU()
).to(config.DEVICE)
policy.fc_sl[0].load_state_dict(policy.fc.state_dict())

# disable grad on sl params (those are frozen, only decoder is trained via SL)
for param in chain(policy.fc_sl.parameters(), policy.policy_head_sl.parameters()):
    param.requires_grad = False


def get_value(self, x):
    return self.value_head(F.relu(self.fc_rl(self.conv_stack(x))))


policy.get_value = partial(get_value, policy)

run, logger = config.wandb_init("lapo_stage3", config.get_wandb_cfg(cfg))

envs = env_utils.setup_procgen_env(
    num_envs=cfg.stage3.num_envs,
    env_id=cfg.env_name,
    gamma=cfg.stage3.gamma,
)

_lr = cfg.stage3.lr
opt, lr_sched = doy.LRScheduler.make(
    torch.optim.Adam,
    policy_head=(
        PLS([0, 35_000, 50_000, cfg.stage3.steps], [0, 0, _lr / 4, 0]),
        [policy.policy_head_rl],
    ),
    value_head=(
        PLS([0, 25_000, 50_000, cfg.stage3.steps], [0, 0, _lr / 4, 0]),
        [policy.value_head],
    ),
    linear_layer=(
        PLS([0, 50_000, 200_000, cfg.stage3.steps], [0, 0, _lr / 10, 0]),
        [policy.fc_rl],
    ),
)

buf_obs = deque(maxlen=3)
buf_la = []
buf_ta = []


def action_selection_hook(next_obs: torch.Tensor, global_step: int = None, action=None):
    # sample action
    hidden_base = policy.conv_stack(next_obs)

    hidden_rl = F.relu(policy.fc_rl(hidden_base))
    hidden_sl = F.relu(policy.fc_sl(hidden_base))

    logits_rl = policy.policy_head_rl(hidden_rl)
    logits_sl = policy.decoder(policy.policy_head_sl(hidden_sl))
    logits = (logits_rl + logits_sl) / 2
    probs = Categorical(logits=logits)

    action_given = action is not None

    if not action_given:
        action = probs.sample()

    if not action_given:
        # update sl-dec data buffer
        buf_obs.append(next_obs.unsqueeze(1))
        if len(buf_obs) == 3:
            buf_la.append(idm(torch.cat(list(buf_obs), dim=1))[0]["la"])
        buf_ta.append(action)

    return action, probs.log_prob(action), probs.entropy(), policy.value_head(hidden_rl)


def reset_decoder(decoder):
    for layer in decoder.children():
        if isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()
        else:
            assert isinstance(layer, torch.nn.ReLU)


def post_update_hook(update, global_step):
    if 10_000 < global_step < 400_000 and (global_step < 50_000 or update % 20 == 0):
        # do decoder online SL training step
        train_ta = torch.stack(buf_ta)[1:-1].flatten(0, 1)
        train_la = torch.stack(buf_la).flatten(0, 1)

        reset_decoder(policy.decoder)

        decoder_opt = torch.optim.Adam(policy.decoder.parameters())

        dataset = TensorDataset(train_la, train_ta.long())
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        num_epochs = 3
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                decoder_opt.zero_grad()
                outputs = policy.decoder(inputs)

                loss = F.cross_entropy(outputs, labels, label_smoothing=0.05)
                loss.backward()
                decoder_opt.step()
            print(f"[{global_step}] loss @ epoch={epoch}: {loss.item()}")


policy = ppo.train(
    policy,
    opt,
    lr_sched,
    logger,
    cfg.stage3,
    envs,
    post_update_hook=post_update_hook,
    action_selection_hook=action_selection_hook,
)

out_path = paths.get_decoded_policy_path(cfg.exp_name)
torch.save(
    {
        "policy": policy.state_dict(),
        "cfg": cfg,
        "logger": logger,
    },
    out_path,
)