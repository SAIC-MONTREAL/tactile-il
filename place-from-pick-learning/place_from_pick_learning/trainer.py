import os
import time
import sys
import json

import torch
from torch.utils.tensorboard import SummaryWriter

def trainer(
    model,
    train_loader,
    opt,
    opt_epoch,
    get_epoch_vars,
    device,
    save_dir,
    model_id,
    checkpoint_dir,
    n_epoch,
    n_checkpoint_epoch,
    val_loader=None,
    sched=None
):
    """Generic boiler plate code for training."""
    # Store tensorboard data
    writer = SummaryWriter(log_dir=save_dir)
    tb_data = []

    # check to see if we're already done
    info_file = os.path.join(save_dir, "model-info.json")
    if os.path.exists(info_file):
        with open(os.path.join(save_dir, "model-info.json"), 'r') as f:
            info = json.load(f)
        if info['epoch'] >= n_epoch:
            print(f"Training of model at {checkpoint_dir} already complete, {info['epoch']} epochs done, "\
                    f"{n_epoch} requested, exiting.")
            sys.exit(0)

    #XXX: If a checkpoint exists, assume preempted and resume training
    initial_epoch = 0
    if os.path.exists(os.path.join(checkpoint_dir, f"checkpoint.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"checkpoint.pth"), map_location=device)
        model.load_state_dict(checkpoint['model'])
        initial_epoch = checkpoint['epoch']
        opt.load_state_dict(checkpoint['opt'])
        if sched is not None:
            sched.load_state_dict(checkpoint['sched'])
        print(f"Resuming training from checkpoint at epoch {initial_epoch}")

    # Training loop
    try:
        for epoch in range(initial_epoch + 1, n_epoch + 1):
            tic = time.time()

            # Iterate any epoch vars if needed
            epoch_vars = get_epoch_vars(epoch)

            # Train for one epoch
            summary_train = opt_epoch(
                loader=train_loader,
                model=model,
                device=device,
                opt=opt,
                **epoch_vars
            )

            if val_loader is not None:
                with torch.no_grad():
                    summary_val = opt_epoch(
                        loader=val_loader,
                        model=model,
                        device=device,
                        **epoch_vars
                    )

            if sched:
                sched.step()
            epoch_time = time.time() - tic

            print(f"Epoch {epoch}/{n_epoch}, Time per epoch: {epoch_time}: , "\
                  f"Total grad updates: {epoch * len(train_loader)}/{n_epoch * len(train_loader)}")
            print(f"Train: {summary_train}")
            if val_loader is not None:
                print(f"Val: {summary_val}")

            # Store tensorboard data
            for k, v in summary_train.items():
                tb_data.append((f"train/{k}", v, epoch))

            if val_loader is not None:
                for k, v in summary_val.items():
                    tb_data.append((f"val/{k}", v, epoch))

            if epoch % n_checkpoint_epoch == 0:
                # Write tensorboard data at intermittent checkpoints
                for data in tb_data:
                    writer.add_scalar(*data)
                tb_data = []

                # Save model at intermittent checkpoints
                cpt = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch
                }
                if sched is not None:
                    cpt["sched"] = sched.state_dict()
                torch.save(
                    cpt,
                    os.path.join(checkpoint_dir, f"checkpoint.pth")
                )
    finally:
        # Save models
        torch.save(model.state_dict(), os.path.join(save_dir, "model-weights.pth"))
        info = {"epoch": epoch}
        with open(os.path.join(save_dir, "model-info.json"), 'w') as f:
            json.dump(info, f)
        writer.close()
