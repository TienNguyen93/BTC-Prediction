import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import yaml
from utils import io_tools
import pytorch_lightning as pl
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Logging directory.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='cmamba_nv',
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )
    parser.add_argument(
        '--save_checkpoints', 
        default=False,   
        action='store_true',          
    )
    parser.add_argument(
        '--use_volume', 
        default=False,   
        action='store_true',          
    )

    parser.add_argument(
        '--resume_from_checkpoint',
        default=None,
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=200,
    )

    #--------------- NEW: Memory optimization arguments ---------------#  
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of steps to accumulate gradients. Use 2-4 to reduce memory.",
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='16-mixed',  # Mixed precision by default
        help="Training precision. Use '16-mixed' or 'bf16-mixed' for memory savings.",
    )
    parser.add_argument(
        '--checkpoint_every_n_epochs',
        type=int,
        default=5,
        help="Save checkpoint every N epochs.",
    )

    parser.add_argument(
        '--log_every_n_epochs',
        type=int,
        default=10,
        help="Log metrics every N epochs to save memory.",
    )

    args = parser.parse_args()
    return args


def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    path = log_dir + '/hparams.yaml'
    if os.path.exists(path):
        return
    with open(path, 'w') as f:
        yaml.dump(save_dict, f)


def load_model(config, logger_type):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)

    normalize = model_config.get('normalize', False)
    hyperparams = config.get('hyperparams')
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get('params')[key] = hyperparams.get(key)

    model_config.get('params')['logger_type'] = logger_type
    model = io_tools.instantiate_from_config(model_config)
    model.cuda()
    model.train()
    return model, normalize


def find_latest_checkpoint(checkpoint_dir):
    """
    Automatically find the latest checkpoint in the directory.
    Returns the path to last.ckpt if it exists, otherwise None.
    """
    import glob
    import os
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # First check for last.ckpt (PyTorch Lightning's default)
    last_ckpt = os.path.join(checkpoint_dir, 'last.ckpt')
    if os.path.exists(last_ckpt):
        print(f"Found checkpoint: {last_ckpt}")
        return last_ckpt
    
    # Otherwise find the most recent checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        print(f"Found checkpoint: {latest}")
        return latest
    
    return None


if __name__ == "__main__":

    args = get_args()
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    use_volume = args.use_volume

    if not use_volume:
        use_volume = config.get('use_volume')
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    model, normalize = load_model(config, args.logger_type)

    tmp = vars(args)
    tmp.update(config)

    name = config.get('name', args.expname)
    if args.logger_type == 'tb':
        logger = TensorBoardLogger("logs", name=name)
        logger.log_hyperparams(args)
    elif args.logger_type == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.expname, config=tmp)
    else:
        raise ValueError('Unknown logger type.')

    data_module = CMambaDataModule(data_config,
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform,
                                   batch_size=args.batch_size,
                                   distributed_sampler=True,
                                   num_workers=args.num_workers,
                                   normalize=normalize,
                                   window_size=model.window_size,
                                   )
    
    callbacks = []
    if args.save_checkpoints:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/rmse",
            mode="min",
            filename='epoch{epoch}-val-rmse{val/rmse:.4f}',
            auto_insert_metric_name=False,
            save_last=True,
            every_n_epochs=args.checkpoint_every_n_epochs,  # Save every N epochs
        )
        callbacks.append(checkpoint_callback)

     # NEW: Custom callback to reduce logging frequency
    class PeriodicLoggingCallback(pl.Callback):
        """Only log metrics every N epochs to save memory"""
        def __init__(self, log_every_n_epochs=10):
            self.log_every_n_epochs = log_every_n_epochs
        
        def on_train_epoch_end(self, trainer, pl_module):
            # Only log on specific epochs
            if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
                # Clear logs to prevent memory accumulation
                if hasattr(trainer, 'logged_metrics'):
                    trainer.logged_metrics.clear()
        
        def on_validation_epoch_end(self, trainer, pl_module):
            current_epoch = trainer.current_epoch + 1
            if current_epoch % self.log_every_n_epochs == 0:
                print(f"\n📊 Epoch {current_epoch}/{trainer.max_epochs} - Metrics logged")
    
    #NEW: Instantiate and add the periodic logging callback
    periodic_logger = PeriodicLoggingCallback(log_every_n_epochs=args.log_every_n_epochs)
    callbacks.append(periodic_logger)
    
    # NEW: Add progress bar for better monitoring
    progress_bar = pl.callbacks.RichProgressBar()
    callbacks.append(progress_bar)


    # CURRENT
    max_epochs = config.get('max_epochs', args.max_epochs)
    model.set_normalization_coeffs(data_module.factors)

    # NEW: Auto-resume logic
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == 'auto':
        # Automatically find the latest checkpoint
        checkpoint_dir = os.path.join('logs', name, 'checkpoints')
        resume_ckpt = find_latest_checkpoint(checkpoint_dir)
        if resume_ckpt:
            print(f"🔄 Auto-resuming from: {resume_ckpt}")
        else:
            print("🆕 No checkpoint found. Starting fresh training.")
            resume_ckpt = None
    elif resume_ckpt:
        print(f"🔄 Resuming from specified checkpoint: {resume_ckpt}")

    trainer = pl.Trainer(accelerator=args.accelerator, 
                         devices=args.devices,
                         max_epochs=max_epochs,
                         enable_checkpointing=args.save_checkpoints,
                         log_every_n_steps=50,
                         logger=logger,
                         callbacks=callbacks,
                         strategy = DDPStrategy(find_unused_parameters=False),

                         # NEW: Memory optimization settings
                         precision=args.precision,  # Mixed precision training
                         accumulate_grad_batches=args.gradient_accumulation_steps,  # Gradient accumulation
                         gradient_clip_val=1.0,  # Gradient clipping to prevent exploding gradients
                         gradient_clip_algorithm='norm',
                         check_val_every_n_epoch=args.log_every_n_epochs, 
                         )


    import torch
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

     # Start/resume training
    try:
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
        
        # Test with best checkpoint
        if args.save_checkpoints:
            print(f"Testing with best model: {checkpoint_callback.best_model_path}")
            trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        if args.save_checkpoints:
            print(f"💾 Progress saved. Resume with: --resume_from_checkpoint auto")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if args.save_checkpoints:
            print(f"💾 Progress saved. Resume with: --resume_from_checkpoint auto")
        raise
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # trainer.fit(model, datamodule=data_module)
    # if args.save_checkpoints:
    #     trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
