import config
import data_loader
import doy
import paths
import torch
import utils
from doy import loop

cfg = config.get()
doy.print("[bold green]Running LAPO stage 1 (IDM/FDM training) with config:")
config.print_cfg(cfg)

run, logger = config.wandb_init("lapo_stage1", config.get_wandb_cfg(cfg))

idm, wm = utils.create_dynamics_models(cfg.model)

train_data, test_data = data_loader.load(cfg.env_name)
train_iter = train_data.get_iter(cfg.stage1.bs)
test_iter = test_data.get_iter(128)

opt, lr_sched = doy.LRScheduler.make(
    all=(
        doy.PiecewiseLinearSchedule(
            [0, 50, cfg.stage1.steps + 1],
            [0.1 * cfg.stage1.lr, cfg.stage1.lr, 0.01 * cfg.stage1.lr],
        ),
        [wm, idm],
    ),
)


def train_step():
    idm.train()
    wm.train()

    lr_sched.step(step)

    batch = next(train_iter)

    vq_loss, vq_perp = idm.label(batch)
    wm_loss = wm.label(batch)
    loss = wm_loss + vq_loss

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters()], 2)
    opt.step()

    logger(
        step,
        wm_loss=wm_loss,
        global_step=step * cfg.stage1.bs,
        vq_perp=vq_perp,
        vq_loss=vq_loss,
        grad_norm=grad_norm,
        **lr_sched.get_state(),
    )


def test_step():
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    # evaluate IDM + FDM generalization on (action-free) test data
    batch = next(test_iter)
    idm.label(batch)
    wm_loss = wm.label(batch)

    # train latent -> true action decoder and evaluate its predictiveness
    _, eval_metrics = utils.eval_latent_repr(train_data, idm)

    logger(step, wm_loss_test=wm_loss, global_step=step * cfg.stage1.bs, **eval_metrics)


for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM"):
    train_step()

    if step % 500 == 0:
        test_step()

    if step > 0 and (step % 5_000 == 0 or step == cfg.stage1.steps):
        torch.save(
            dict(
                **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                step=step,
                cfg=cfg,
                logger=logger,
            ),
            paths.get_models_path(cfg.exp_name),
        )
