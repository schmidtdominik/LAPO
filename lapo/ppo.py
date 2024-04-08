import time

import config
import doy
import numpy as np
import torch
import torch.nn as nn
from data_loader import normalize_obs
from tensordict import TensorDict


def create_buffer(
    num_steps: int, num_envs: int, obs_space, action_space, device
) -> TensorDict:
    return TensorDict(
        {
            "obs": torch.zeros(
                (num_steps, num_envs, obs_space.shape[2], *obs_space.shape[:2]),
                dtype=torch.uint8,
            ),
            "actions": torch.zeros(
                (num_steps, num_envs, *action_space.shape), dtype=torch.long
            ),
            "logprobs": torch.zeros((num_steps, num_envs)),
            "rewards": torch.zeros((num_steps, num_envs)),
            "dones": torch.zeros((num_steps, num_envs)),
            "values": torch.zeros((num_steps, num_envs)),
            # "adv": None,
            # "returns": None,
        },
        batch_size=num_steps,
        device=device,
    )


def _batch_update(
    policy,
    batch: TensorDict,
    rl_cfg: config.Stage3Config,
    loss_scale,
    action_selection_hook,
):
    if action_selection_hook is not None:
        _, newlogprob, entropy, newvalue = action_selection_hook(
            normalize_obs(batch["obs"]), action=batch["actions"]
        )
    else:
        _, newlogprob, entropy, newvalue = policy.get_action_and_value(
            normalize_obs(batch["obs"]), action=batch["actions"]
        )
    logratio = newlogprob - batch["logprobs"]
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs = ((ratio - 1.0).abs() > rl_cfg.clip_coef).float().mean().item()

    if rl_cfg.norm_adv:
        batch["adv"] = (batch["adv"] - batch["adv"].mean()) / (
            batch["adv"].std() + 1e-8
        )

    # Policy loss
    pg_loss1 = -batch["adv"] * ratio
    pg_loss2 = -batch["adv"] * torch.clamp(
        ratio, 1 - rl_cfg.clip_coef, 1 + rl_cfg.clip_coef
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if rl_cfg.clip_vloss:
        v_loss_unclipped = (newvalue - batch["returns"]) ** 2
        v_clipped = batch["values"] + torch.clamp(
            newvalue - batch["values"],
            -rl_cfg.clip_coef,
            rl_cfg.clip_coef,
        )
        v_loss_clipped = (v_clipped - batch["returns"]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - batch["returns"]) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - rl_cfg.ent_coef * entropy_loss + v_loss * rl_cfg.vf_coef

    (loss / loss_scale).backward()

    return approx_kl, clipfracs, pg_loss, v_loss, entropy_loss


def batch_update(policy, opt, batch: TensorDict, cfg, action_selection_hook):
    """do PPO update on given batch, optionally with grad accumulation"""

    opt.zero_grad()
    batches = batch.chunk(cfg.grad_accum_f)
    assert len(batches) == cfg.grad_accum_f

    results = []
    for batch in batches:
        results.append(
            _batch_update(
                policy,
                batch,
                cfg,
                loss_scale=cfg.grad_accum_f,
                action_selection_hook=action_selection_hook,
            )
        )

    nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    opt.step()

    return [sum(x) / len(x) for x in zip(*results)]


def _bootstrap(policy, buf, next_obs, next_done, rlft_cfg):
    # bootstrap value if not done
    with torch.no_grad():
        next_value = policy.get_value(normalize_obs(next_obs)).reshape(1, -1)
        buf["adv"] = torch.zeros_like(buf["rewards"])  # mem leak?

        lastgaelam = 0
        for t in reversed(range(rlft_cfg.num_steps)):
            if t == rlft_cfg.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - buf["dones"][t + 1]
                nextvalues = buf["values"][t + 1]
            delta = (
                buf["rewards"][t]
                + rlft_cfg.gamma * nextvalues * nextnonterminal
                - buf["values"][t]
            )
            buf["adv"][t] = lastgaelam = (
                delta
                + rlft_cfg.gamma * rlft_cfg.gae_lambda * nextnonterminal * lastgaelam
            )
        buf["returns"] = buf["adv"] + buf["values"]


def _update(
    policy,
    opt,
    lr_sched,
    logger,
    rl_cfg,
    buf,
    global_step,
    start_time,
    action_selection_hook,
):
    # flatten the batch
    buf_flat = buf.flatten(0, 1)

    # Optimizing the policy and value network
    inds = np.arange(rl_cfg.batch_size)
    clipfracs = []
    for _ in range(rl_cfg.update_epochs):
        np.random.shuffle(inds)

        for start in range(0, rl_cfg.batch_size, rl_cfg.minibatch_size):
            end = start + rl_cfg.minibatch_size
            mb_inds = inds[start:end]
            batch: TensorDict = buf_flat[mb_inds]  # type: ignore
            (
                approx_kl,
                clipfracs,
                pg_loss,
                v_loss,
                entropy_loss,
            ) = batch_update(policy, opt, batch, rl_cfg, action_selection_hook)

        if rl_cfg.target_kl is not None:
            if approx_kl > rl_cfg.target_kl:
                break

    y_pred, y_true = (
        buf_flat["values"].cpu().numpy(),
        buf_flat["returns"].cpu().numpy(),
    )
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    logger(
        step=global_step,
        global_step=global_step,
        value_loss=v_loss,
        policy_loss=pg_loss,
        entropy=entropy_loss,
        approx_kl=approx_kl,
        clipfrac=np.mean(clipfracs),
        explained_variance=explained_var,
        SPS=int(global_step / (time.time() - start_time)),
        **lr_sched.get_state(),
    )


def train(
    policy,
    opt,
    lr_sched,
    logger: doy.Logger,
    rl_cfg,
    envs,
    post_update_hook=lambda *_: None,
    action_selection_hook=None,
):
    device = config.DEVICE

    buf = create_buffer(
        rl_cfg.num_steps,
        rl_cfg.num_envs,
        envs.single_observation_space,
        envs.single_action_space,
        device,
    )

    global_step = 0
    start_time = time.time()
    next_obs = torch.from_numpy(envs.reset()).permute((0, 3, 1, 2)).to(device)
    next_done = torch.zeros(rl_cfg.num_envs).to(device).float()
    num_updates = rl_cfg.steps // rl_cfg.batch_size

    for update in doy.loop(1, num_updates + 1, desc="Running PPO training..."):
        if rl_cfg.anneal_lr:
            lr_sched.step(global_step)
        else:
            assert (
                False
            ), "this is prob broken when anneal_lr is False (since we're using an lr_sched)"

        for step in range(0, rl_cfg.num_steps):
            buf["obs"][step] = next_obs
            buf["dones"][step] = next_done

            # [acting]
            with torch.no_grad():
                if action_selection_hook:
                    action, buf["logprobs"][step], _, value = action_selection_hook(
                        normalize_obs(next_obs), global_step
                    )
                else:
                    action, buf["logprobs"][step], _, value = (
                        policy.get_action_and_value(normalize_obs(next_obs))
                    )

                buf["values"][step] = value.flatten()
            buf["actions"][step] = action

            # [env.step]
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            buf["rewards"][step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.from_numpy(next_obs).permute((0, 3, 1, 2)).to(device)
            next_done = torch.from_numpy(done).to(device).float()

            for substep, item in enumerate(info):
                if "episode" in item.keys():
                    logger(
                        step=global_step + substep,
                        global_step=global_step + substep,
                        episodic_return=item["episode"]["r"],
                        episodic_length=item["episode"]["l"],
                        episodic_return_norm=envs.normalize_return(
                            item["episode"]["r"]
                        ),
                    )
                    break
            global_step += 1 * rl_cfg.num_envs

        _bootstrap(policy, buf, next_obs, next_done, rl_cfg)
        _update(
            policy,
            opt,
            lr_sched,
            logger,
            rl_cfg,
            buf,
            global_step,
            start_time,
            action_selection_hook,
        )

        post_update_hook(update, global_step)
    return policy