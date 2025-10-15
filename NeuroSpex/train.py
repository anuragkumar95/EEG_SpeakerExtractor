import os
import wandb
import json
import argparse
import torch
from Data.dataset import EEGDataset, NeurHeedEEG_Dataset
from Models.neurospex import NeuroSpex
from utils import SISDRLoss, custom_collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

### TODO: Add multi-GPU training.

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    #os.environ["MASTER_ADDR"] = "127.0.0.1"
    #os.environ["MASTER_PORT"] = "8889"
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    init_process_group(backend, rank=rank, world_size=world_size)

class Trainer:
    def __init__(self, train_ds, val_ds, config, log_wandb=True, gpu_id=0, parallel=False):
        self.config = config
        self.gpu = gpu_id if parallel else config['gpu']

        # Initialize model
        self.model = NeuroSpex(
            self.config['speech_encoder_params'],
            self.config['eeg_encoder_params'],
            self.config['speech_decoder_params'],
            self.config['spk_ext_params']
        )

        self.optimizer = torch.optim.Adam(
                filter(lambda layer:layer.requires_grad, self.model.parameters()), lr=config["init_lr"]
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )

        if "resume_pt" in config and config["resume_pt"] is not None:
            self.load(config["resume_pt"])

        self.model = self.model.to(gpu_id)
        if parallel:
            self.model = DDP(self.model, device_ids=[gpu_id])
       
        self.loss_fn = SISDRLoss()
        self.accum_grad = config['accum_grad']
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.log_wandb = False
        if log_wandb and gpu_id == 0:
            self.log_wandb = True
            wandb.login()
            wandb.init(project=config["experiment"], name=config["run_name"])

    def save(self, checkpoint_path, epoch):
        if self.gpu == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    def load(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Checkpoint loaded successfully.")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    def forward_step(self, batch):
        mixture, target, eeg = batch
        estimated = self.model(mixture, eeg)
        return estimated

    def train_one_step(self, step, batch):
        _, tgt, _ = batch
        est = self.forward_step(batch)

        est = est.squeeze(1)
        tgt = tgt.squeeze(1)
        
        loss = self.loss_fn(est, tgt) / self.accum_grad
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        if (step + 1) % self.accum_grad == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss * self.accum_grad

    def validate_one_step(self, step, batch):
        _, tgt, _ = batch
        est = self.forward_step(batch)
        est = est.squeeze(1)
        loss = self.loss_fn(est, tgt)
        return loss

    def validate(self, ep, step):
        ############################## VALIDATION ##############################
        val_loss = 0.0
        with torch.no_grad():
            step = 0
            pbar = tqdm(self.val_ds)
            for batch in pbar:
                # Set the right device
                mixture, target, eeg = batch
                if self.gpu is not None:
                    mixture = mixture.to(self.gpu)
                    target = target.to(self.gpu)
                    eeg = eeg.to(self.gpu)
                batch = (mixture, target, eeg)
                
                loss = self.validate_one_step(step, batch)
                val_loss += loss.item()

                avg_loss_so_far = val_loss / (step + 1)

                pbar.set_postfix({
                    'Loss': avg_loss_so_far
                })
                step += 1
                
        avg_val_loss = val_loss / len(self.val_ds)
        return avg_val_loss
        
    def train(self, epochs):
        best_val_loss = float('inf')
        no_improv_counter = 0
        for ep in range(epochs):
            self.model.train()
            ############################# TRAINING ##############################
            total_loss = 0.0
            step = 0
            if self.gpu == 0:
                pbar = tqdm(self.train_ds)
            else:
                pbar = self.train_ds
            for batch in pbar:
                # Set the right device
                mixture, target, eeg = batch
                if self.gpu is not None:
                    mixture = mixture.to(self.gpu)
                    target = target.to(self.gpu)
                    eeg = eeg.to(self.gpu)
                
                batch = (mixture, target, eeg)
                loss = self.train_one_step(step, batch)
                total_loss += loss.item()
                if self.log_wandb:
                    wandb.log({
                        "step": (ep + 1) * len(self.train_ds) + (step+1),
                        "train/loss": loss.item()
                    })

                avg_loss_so_far = total_loss / (step + 1)

                if self.gpu == 0:
                    pbar.set_postfix({
                        'Epoch': ep + 1,
                        'Loss': avg_loss_so_far
                    })

                if (step+1) % self.config['val_every_step'] == 0:
                    self.model.eval()
                    val_loss = self.validate(ep, step)
                    if self.log_wandb:
                        wandb.log({
                            "step": (ep + 1) * len(self.train_ds) + (step+1),
                            "val/loss": val_loss
                        })
                    self.model.train()
                step += 1
            
            avg_loss = total_loss / len(self.train_ds)
            print(f"Epoch [{ep+1}/{epochs}] Training Loss: {avg_loss:.4f}")
            self.model.eval()
            val_loss = self.validate(ep, step)
            if self.log_wandb:
                wandb.log({
                    "step": (ep + 1) * len(self.train_ds) + (step+1),
                    "train/ep_loss": avg_loss,
                    "epoch": ep + 1,
                    "val/loss": val_loss
                })

            if best_val_loss > val_loss:
                # Save checkpoint
                checkpoint_dir = f"{self.config['save_dir']}/{self.config['experiment']}/{self.config['run_name']}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{ep+1}.pth")
                self.save(checkpoint_path, ep+1)
                best_val_loss = val_loss
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                no_improv_counter = 0
            else:
                # Early stopping and LR scheduling
                no_improv_counter += 1
                if no_improv_counter % 25 == 0:
                    print("No improvement for 25 epochs, stopping training.")
                    break
                elif no_improv_counter % 5 == 0:
                    self.scheduler.step()
                    print("No improvement for 5 epochs, reducing learning rate.")
        
        
def main(rank, world_size, ARGS):
    if ARGS.parallel:
        ddp_setup(rank, world_size)
        if rank == 0:
            available_gpus = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            print(f"Available gpus:{available_gpus}")

    # Load config
    with open(ARGS.config, 'r') as f:
        config = json.load(f)

    data_config = config["data"]
    train_config = config["train"]

    # Prepare datasets
    #train_dataset = EEGDataset(root=data_config["root"], split='train')
    train_dataset = NeurHeedEEG_Dataset(
        root=data_config["root"], 
        partition='train', 
        batch_size=train_config["batch_size"],
        max_length=8, 
        audio_sr=8000, 
        ref_sr=128)

    #val_dataset = EEGDataset(root=data_config["root"], split='val')
    val_dataset = NeurHeedEEG_Dataset(
        root=data_config["root"], 
        partition='val',
        batch_size=train_config["batch_size"], 
        max_length=8, 
        audio_sr=8000, 
        ref_sr=128)

    print(f"TRAIN:{len(train_dataset)} | VAL:{len(val_dataset)}")
    print(f"GPU: {train_config['gpu']} | BATCH SIZE: {train_config['batch_size']} | ACCUM GRAD: {train_config['accum_grad']}")

    # ONLY USE IF USING NEUROHEED DATASET

    train_loader = DataLoader(
        train_dataset, 
        #batch_size=train_config["batch_size"], 
        batch_size=1,
        sampler=DistributedSampler(train_dataset),
        shuffle=False, 
        num_workers=train_config["batch_size"],
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        #batch_size=train_config["batch_size"], 
        batch_size=1,
        sampler=DistributedSampler(val_dataset),
        shuffle=False, 
        num_workers=train_config["batch_size"],
        collate_fn=custom_collate_fn
    )

    # Initialize trainer
    trainer = Trainer(
        train_ds=train_loader, 
        val_ds=val_loader, 
        config=train_config, 
        log_wandb=ARGS.logwandb,
        gpu_id=rank if ARGS.parallel else train_config['gpu'],
        parallel=ARGS.parallel
    )

    print(f"Starting training for {train_config['epochs']} epochs...")
    # Start training
    trainer.train(epochs=train_config["epochs"])
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--logwandb', action='store_true', help='Enable logging to WandB')
    parser.add_argument('--parallel', action='store_true', help='Enable multi-GPU training.')
    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    if args.parallel:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        main(None, world_size, args)