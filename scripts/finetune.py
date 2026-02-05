import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

import copy
import types

import lightning as pl
from imscore.hps.model import HPSv2
from imscore.imreward.model import ImageReward
from imscore.pickscore.model import PickScorer
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from mfm.data import get_data_module
from mfm.losses import get_finetuning_loss_fn
from mfm.models.model_wrapper import SIModelWrapper
from mfm.utils import EMAWeightAveraging, FinetuningModule, RewardCallback


@hydra.main(
    config_path="../conf/", config_name="config_finetune.yaml", version_base="1.3"
)
def main(cfg: DictConfig):

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.get("entity", None),
        config=dict(cfg),
    )
    OmegaConf.set_struct(cfg, False)
    log_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    cfg.work_dir = log_dir
    seed_everything(cfg.seed, workers=True)

    # load in mfm
    mfm = instantiate(cfg.model)
    SI = instantiate(cfg.SI)
    mfm = SIModelWrapper(mfm, SI, cfg.use_parametrization)

    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")

    checkpoint = {
        k[6:] if k.startswith("model.") else k: v for k, v in checkpoint.items()
    }
    missing, unexpected = mfm.load_state_dict(checkpoint, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    model = copy.deepcopy(mfm)
    
    # extract the unconditional dynamics to fine-tune
    def custom_forward(self, x, t, y, **kwargs):
        return self.v(
            t, t, x, torch.zeros_like(t), torch.zeros_like(x), class_labels=y, **kwargs
        )

    model.forward = types.MethodType(custom_forward, model)

    # load in data
    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()
    inverse_scaler = datamodule.inverse_scaler
    loss_fn = get_finetuning_loss_fn(cfg, SI)
    base_model = copy.deepcopy(model)

    # load in reward model
    if cfg.get("imagereward_function", "HPSv2") == "HPSv2":
        reward_fn = HPSv2.from_pretrained("RE-N-Y/hpsv21") 
    elif cfg.get("imagereward_function") == "ImageReward":
        reward_fn = ImageReward.from_pretrained("RE-N-Y/ImageReward")
    elif cfg.get("imagereward_function") == "PickScore":
        reward_fn = PickScorer.from_pretrained("RE-N-Y/pickscore")

    reward_fn.eval()
    
    # additional models for evaluation
    evaluation_reward_models = {
        "ImageReward": ImageReward.from_pretrained("RE-N-Y/ImageReward"),
        "PickScore": PickScorer.from_pretrained("RE-N-Y/pickscore"),
        "HPSv2": HPSv2.from_pretrained("RE-N-Y/hpsv21"),
    }

    train_module = FinetuningModule(
        cfg,
        model,
        base_model,
        mfm,
        reward_fn,
        loss_fn,
        SI,
        eval_reward_models=evaluation_reward_models,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoints",
        filename="periodic-{epoch:02d}-{step}",
        every_n_train_steps=cfg.trainer.get("checkpoint_every_n_steps", 10000),
        save_top_k=-1,
    )
    reward_callback = RewardCallback(cfg)
    ema_callback = EMAWeightAveraging(
        cfg.trainer.ema.decay, min_step=cfg.trainer.ema.min_step
    )
    callbacks = [ema_callback, checkpoint_callback, reward_callback]

    enable_progress_bar = True
    if "SLURM_PROCID" in os.environ and int(os.environ["SLURM_PROCID"]) > 0:
        enable_progress_bar = False

    if cfg.filter_for_classes:
        limit_val = 1.0
    else:
        limit_val = 0.01

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_steps=cfg.trainer.num_train_steps,
        accelerator="gpu",
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.get("num_nodes", 1),
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        enable_progress_bar=enable_progress_bar,
        limit_val_batches=limit_val,
        val_check_interval=cfg.sampling.get("every_n_steps", 500),
        check_val_every_n_epoch=None,
    )

    resume_path = cfg.get("resume_from_checkpoint", None)
    ckpt_path = resume_path
    trainer.fit(train_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
