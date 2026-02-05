import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

import lightning as pl
import torch.nn as nn
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from mfm.data import get_data_module
from mfm.losses import get_consistency_loss_fn
from mfm.models.model_wrapper import SIModelWrapper
from mfm.utils import EMAWeightAveraging, SamplingCallback, TrainingModule

def get_conditioning_data(cfg, datamodule):
    val_dataset = datamodule.imagenet_val
    target_classes = torch.arange(1000)
    indices = [i for i, t in enumerate(val_dataset.targets) if t in target_classes]

    if len(indices) < cfg.sampling.n_conditioning_samples:
        print(f"Warning: Only found {len(indices)} samples. Repeating to fill batch.")
        indices = indices * (cfg.sampling.n_conditioning_samples // len(indices) + 1)

    perm = torch.randperm(len(indices))
    indices = [indices[i] for i in perm[: cfg.sampling.n_conditioning_samples].tolist()]
    test_data = torch.stack([val_dataset[i][0] for i in indices])
    test_labels = torch.tensor([val_dataset[i][1] for i in indices])
    return test_data, test_labels

@hydra.main(config_path="../conf/", config_name="config_train.yaml", version_base="1.3")
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

    model = instantiate(cfg.model)

    SI = instantiate(cfg.SI)
    model = SIModelWrapper(model, SI, cfg.use_parametrization)
    weighting_model = (
        instantiate(cfg.weighting_model) if cfg.model.learn_loss_weighting else None
    )

    datamodule = get_data_module(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    inverse_scaler = datamodule.inverse_scaler
    loss_fn = get_consistency_loss_fn(cfg, SI)
    train_module = TrainingModule(cfg, model, weighting_model, loss_fn, SI)

    if cfg.init_from_dmf:
        sit_state_dict = torch.load(
            cfg.dmf_path, map_location="cpu", weights_only=False
        )
        sit_state_dict = {
            k.replace("module.", ""): v for k, v in sit_state_dict.items()
        }

        # Target model (DiT)
        target_model = train_module.model.model
        if hasattr(target_model, "dit"):
            target_model = target_model.dit

        # Load state dict
        print("Loading state dict into DiT...")
        missing, unexpected = target_model.load_state_dict(sit_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        # Initialize s_embedder from t_embedder (checkpoint)
        print("Initializing s_embedder from t_embedder...")
        target_model.s_embedder.load_state_dict(target_model.t_embedder.state_dict())
        target_model.t_embedder_second.load_state_dict(
            target_model.t_embedder.state_dict()
        )
        print("Initializing x_cond_embedder from x_embedder...")
        target_model.x_cond_embedder.load_state_dict(
            target_model.x_embedder.state_dict()
        )

        # zero t_embedder
        nn.init.constant_(target_model.t_embedder.mlp[2].weight, 0)
        nn.init.constant_(target_model.t_embedder.mlp[2].bias, 0)

        # Sync teacher model with student model after loading weights
        if train_module.teacher_model is not None:
            print("Syncing teacher model weights from student...")
            train_module.teacher_model.load_state_dict(train_module.model.state_dict())

    # callbacks
    ema_callback = EMAWeightAveraging(cfg.trainer.ema.decay)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoints",
        filename="periodic-{epoch:02d}-{step}",
        every_n_train_steps=cfg.trainer.get("checkpoint_every_n_steps", 10000),
        save_top_k=-1,
    )

    test_data, test_labels = get_conditioning_data(cfg, datamodule)
    sampling_callback = SamplingCallback(
        cfg, test_data, test_labels, inverse_scaler, SI
    )
    callbacks = [ema_callback, sampling_callback, checkpoint_callback]

    # Enable progress bar only for rank 0 in distributed training
    if "SLURM_PROCID" in os.environ and int(os.environ["SLURM_PROCID"]) > 0:
        enable_progress_bar = False
    else:
        enable_progress_bar = True

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_steps=cfg.trainer.num_train_steps,
        accelerator="gpu",
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=cfg.model.learn_loss_weighting),
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        enable_progress_bar=enable_progress_bar,
    )

    resume_path = cfg.get("resume_from_checkpoint", None)
    ckpt_path = resume_path
    trainer.fit(train_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
