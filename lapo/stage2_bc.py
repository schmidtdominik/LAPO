import config
import data_loader
import doy
import paths
import torch
import torch.nn.functional as F
import utils
from doy import loop

state_dicts = torch.load(paths.get_models_path(config.get().exp_name))
cfg = config.get(base_cfg=state_dicts["cfg"], reload_keys=["stage2", "stage3"])
cfg.stage_exp_name = doy.random_proquint(1)
doy.print("[bold green]Running LAPO stage 2 (latent behavior cloning) with config:")
config.print_cfg(cfg)

if state_dicts["step"] != cfg.stage1.steps:
    doy.log(
        f"[bold red]Warning: using IDM/WM from incomplete training run {state_dicts['step']}/{cfg.stage1.steps} steps"
    )

idm, _ = utils.create_dynamics_models(cfg.model, state_dicts=state_dicts)
idm.eval()

policy = utils.create_policy(cfg.model, cfg.model.la_dim)
opt, lr_sched = doy.LRScheduler.make(
    policy=(
        doy.PiecewiseLinearSchedule(
            [0, 1000, cfg.stage2.steps + 1], [0.01 * cfg.stage2.lr, cfg.stage2.lr, 0]
        ),
        [policy],
    ),
)

train_data, test_data = data_loader.load(cfg.env_name)
train_iter = train_data.get_iter(cfg.stage2.bs)
test_iter = test_data.get_iter(128)

_, eval_metrics = utils.eval_latent_repr(train_data, idm)
doy.log(f"Decoder metrics sanity check: {eval_metrics}")

run, logger = config.wandb_init("lapo_stage2", config.get_wandb_cfg(cfg))

for step in loop(
    cfg.stage2.steps + 1, desc="[green bold](stage-2) Training latent policy via BC"
):
    lr_sched.step(step)

    policy.train()
    batch = next(train_iter)
    idm.label(batch)

    preds = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
    loss = F.mse_loss(preds, batch["la"])

    opt.zero_grad()
    loss.backward()
    opt.step()

    logger(
        step=step,
        loss=loss,
        **lr_sched.get_state(),
    )

    if step % 200 == 0:
        policy.eval()
        test_batch = next(test_iter)
        idm.label(test_batch)
        test_loss = F.mse_loss(policy(test_batch["obs"][:, -2]), test_batch["la"])
        logger(step=step, test_loss=test_loss)

torch.save(
    dict(policy=doy.state_dict_orig(policy), cfg=cfg, logger=logger),
    paths.get_latent_policy_path(cfg.exp_name),
)
