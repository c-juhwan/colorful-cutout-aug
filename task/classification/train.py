# Standard Library Modules
import os
import sys
import shutil
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.model import ClassificationModel
from model.classification.dataset import CustomDataset, collate_fn
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path, get_cutout_box

def training(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load dataset and define dataloader
    write_log(logger, "Loading data")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = CustomDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'train_processed.pkl'))
    dataset_dict['valid'] = CustomDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'valid_processed.pkl'))

    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    args.num_classes = dataset_dict['train'].num_classes

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")
    write_log(logger, f"Valid dataset size / iterations: {len(dataset_dict['valid'])} / {len(dataloader_dict['valid'])}")

    # Get model instance
    write_log(logger, "Building model")
    model = ClassificationModel(args).to(device)

    # Define optimizer and scheduler
    write_log(logger, "Building optimizer and scheduler")
    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, optim_type=args.optimizer)
    scheduler = get_scheduler(optimizer, len(dataloader_dict['train']), num_epochs=args.num_epochs,
                              early_stopping_patience=args.early_stopping_patience, learning_rate=args.learning_rate,
                              scheduler_type=args.scheduler)
    write_log(logger, f"Optimizer: {optimizer}")
    write_log(logger, f"Scheduler: {scheduler}")

    # Define loss function
    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps)
    write_log(logger, f"Loss function: {cls_loss}")

    # If resume_training, load from checkpoint
    start_epoch = 0
    if args.job == 'resume_training':
        write_log(logger, "Resuming training model")
        load_checkpoint_name = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type,
                                            f'checkpoint.pt')
        model = model.to('cpu')
        checkpoint = torch.load(load_checkpoint_name, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.to(device)
        write_log(logger, f"Loaded checkpoint from {load_checkpoint_name}")

        if args.use_wandb:
            import wandb # Only import wandb when it is used
            from wandb import AlertLevel
            wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Model: {args.model_type}",
                             f"Aug: {args.augmentation_type}"],
                       resume=True,
                       id=checkpoint['wandb_id'])
            wandb.watch(models=model, criterion=cls_loss, log='all', log_freq=10)
        del checkpoint

    # Initialize tensorboard writer
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Initialize wandb
    if args.use_wandb and args.job == 'training':
        import wandb # Only import wandb when it is used
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Model: {args.model_type}",
                             f"Aug: {args.augmentation_type}"])
        wandb.watch(models=model, criterion=cls_loss, log='all', log_freq=10)

    # Train/Valid - Start training
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    write_log(logger, f"Start training from epoch {start_epoch}")
    for epoch_idx in range(start_epoch, args.num_epochs):
        # Train - Set model to train mode
        model = model.train()
        train_loss_cls = 0
        train_acc_cls = 0
        train_f1_cls = 0

        # Train - Iterate one epoch
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            # Train - Get input data
            images = data_dicts['images'].to(device)
            labels = data_dicts['labels'].to(device)

            if args.augmentation_type == 'none':
                classification_logits = model(images)
                batch_loss_cls = cls_loss(classification_logits, labels)
            elif args.augmentation_type == 'cutout':
                bx1, bx2, by1, by2 = get_cutout_box(args.image_crop_size, args.augmentation_box_size)
                masked_images = images.clone()
                masked_images[:, :, bx1:bx2, by1:by2] = 0 # Mask out the cutout region

                classification_logits = model(masked_images)
                batch_loss_cls = cls_loss(classification_logits, labels)
            elif args.augmentation_type == 'color_cutout_nocur':
                bx1, bx2, by1, by2 = get_cutout_box(args.image_crop_size, args.augmentation_box_size)
                masked_images = images.clone()
                mask_color = torch.rand(images.shape[0], images.shape[1], 1, 1).to(device) # Generate random color for each image
                masked_images[:, :, bx1:bx2, by1:by2] = mask_color # Mask out the cutout region with random color

                classification_logits = model(masked_images)
                batch_loss_cls = cls_loss(classification_logits, labels)
            elif args.augmentation_type == 'color_cutout_cur':
                bx1, bx2, by1, by2 = get_cutout_box(args.image_crop_size, args.augmentation_box_size)
                masked_images = images.clone()

                # Generate random color for each image
                region_size = args.augmentation_box_size // (2 ** epoch_idx) # Decrease the region size as epoch increases
                region_amount = args.augmentation_box_size // region_size # We have region_amount ** 2 regions

                for i in range(region_amount ** 2): # We have to generate random color for each region
                    rx1 = bx1 + region_size * (i // region_amount) # Region x1
                    rx2 = bx1 + region_size * (i // region_amount + 1) # Region x2
                    ry1 = by1 + region_size * (i % region_amount) # Region y1
                    ry2 = by1 + region_size * (i % region_amount + 1) # Region y2

                    mask_color = torch.rand(images.shape[0], images.shape[1], 1, 1).to(device)
                    masked_images[:, :, rx1:rx2, ry1:ry2] = mask_color # Mask out the cutout region with random color

                classification_logits = model(masked_images)
                batch_loss_cls = cls_loss(classification_logits, labels)
            elif args.augmentation_type == 'mixup':
                # Generate mixup ratio
                mixup_ratio = torch.distributions.beta.Beta(args.augmentation_mixup_alpha, args.augmentation_mixup_alpha).sample().to(device)
                mixup_ratio = torch.max(mixup_ratio, 1 - mixup_ratio)

                # Generate mixup images
                mixup_images = images.clone()
                target_idx = torch.randperm(images.shape[0]).to(device)

                mixup_images = mixup_ratio.view(-1, 1, 1, 1) * mixup_images + (1 - mixup_ratio).view(-1, 1, 1, 1) * mixup_images[target_idx, :, :, :]
                classification_logits = model(mixup_images)
                batch_loss_cls = mixup_ratio * cls_loss(classification_logits, labels) + (1 - mixup_ratio) * cls_loss(classification_logits, labels[target_idx])
            elif args.augmentation_type == 'cutmix':
                # Generate mixup ratio
                mixup_ratio = torch.distributions.beta.Beta(args.augmentation_mixup_alpha, args.augmentation_mixup_alpha).sample().to(device)
                mixup_ratio = torch.max(mixup_ratio, 1 - mixup_ratio)

                # Generate mixup images
                mixup_images = images.clone()
                target_idx = torch.randperm(images.shape[0]).to(device)
                bx1, bx2, by1, by2 = get_cutout_box(args.image_crop_size, args.augmentation_box_size)

                mixup_images[:, :, bx1:bx2, by1:by2] = mixup_images[target_idx, :, bx1:bx2, by1:by2]
                mixup_lambda = 1 - ((bx2 - bx1) * (by2 - by1) / (mixup_images.shape[2] * mixup_images.shape[3]))

                classification_logits = model(mixup_images)
                batch_loss_cls = mixup_lambda * cls_loss(classification_logits, labels) + (1 - mixup_lambda) * cls_loss(classification_logits, labels[target_idx])
            else:
                raise NotImplementedError(f'Invalid augmentation type: {args.augmentation_type}')

            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            # Train - Backward pass
            optimizer.zero_grad()
            batch_loss_cls.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step() # These schedulers require step() after every training iteration

            # Train - Logging
            train_loss_cls += batch_loss_cls.item()
            train_acc_cls += batch_acc_cls.item()
            train_f1_cls += batch_f1_cls

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['train']) - 1:
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Loss: {batch_loss_cls.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Acc: {batch_acc_cls.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - F1: {batch_f1_cls:.4f}")
            if args.use_tensorboard:
                writer.add_scalar('TRAIN/Learning_Rate', optimizer.param_groups[0]['lr'], epoch_idx * len(dataloader_dict['train']) + iter_idx)

        # Train - End of epoch logging
        if args.use_tensorboard:
            writer.add_scalar('TRAIN/Loss', train_loss_cls / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Acc', train_acc_cls / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/F1', train_f1_cls / len(dataloader_dict['train']), epoch_idx)

        # Valid - Set model to eval mode
        model = model.eval()
        valid_loss_cls = 0
        valid_acc_cls = 0
        valid_f1_cls = 0

        # Valid - Iterate one epoch
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc=f'Validating - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            # Valid - Get input data
            images = data_dicts['images'].to(device)
            labels = data_dicts['labels'].to(device)

            # Valid - Forward pass
            with torch.no_grad():
                classification_logits = model(images)

            # Valid - Calculate loss & accuracy/f1 score
            batch_loss_cls = cls_loss(classification_logits, labels)
            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            # Valid - Logging
            valid_loss_cls += batch_loss_cls.item()
            valid_acc_cls += batch_acc_cls.item()
            valid_f1_cls += batch_f1_cls

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['valid']) - 1:
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Loss: {batch_loss_cls.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Acc: {batch_acc_cls.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - F1: {batch_f1_cls:.4f}")

        # Valid - Call scheduler
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_loss_cls)

        # Valid - Check loss & save model
        valid_loss_cls /= len(dataloader_dict['valid'])
        valid_acc_cls /= len(dataloader_dict['valid'])
        valid_f1_cls /= len(dataloader_dict['valid'])

        if args.optimize_objective == 'loss':
            valid_objective_value = valid_loss_cls
            valid_objective_value = -1 * valid_objective_value # Loss is minimized, but we want to maximize the objective value
        elif args.optimize_objective == 'accuracy':
            valid_objective_value = valid_acc_cls
        elif args.optimize_objective == 'f1':
            valid_objective_value = valid_f1_cls
        else:
            raise NotImplementedError

        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"VALID - Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0 # Reset early stopping counter

            checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type)
            check_path(checkpoint_save_path)

            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None
            }, os.path.join(checkpoint_save_path, f'checkpoint.pt'))
            write_log(logger, f"VALID - Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
            write_log(logger, f"VALID - Saved checkpoint to {checkpoint_save_path}")
        else:
            early_stopping_counter += 1
            write_log(logger, f"VALID - Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")

        # Valid - End of epoch logging
        if args.use_tensorboard:
            writer.add_scalar('VALID/Loss', valid_loss_cls, epoch_idx)
            writer.add_scalar('VALID/Acc', valid_acc_cls, epoch_idx)
            writer.add_scalar('VALID/F1', valid_f1_cls, epoch_idx)
        if args.use_wandb:
            wandb.log({'TRAIN/Epoch_Loss': train_loss_cls / len(dataloader_dict['train']),
                        'TRAIN/Epoch_Acc': train_acc_cls / len(dataloader_dict['train']),
                        'TRAIN/Epoch_F1': train_f1_cls / len(dataloader_dict['train']),
                        'VALID/Epoch_Loss': valid_loss_cls,
                        'VALID/Epoch_Acc': valid_acc_cls,
                        'VALID/Epoch_F1': valid_f1_cls,
                        'Epoch_Index': epoch_idx})
            wandb.alert(
                title='Epoch End',
                text=f"VALID - Epoch {epoch_idx} - Loss: {valid_loss_cls:.4f} - Acc: {valid_acc_cls:.4f}",
                level=AlertLevel.INFO,
                wait_duration=300
            )

        # Valid - Early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f"VALID - Early stopping at epoch {epoch_idx}...")
            break

    # Final - End of training
    write_log(logger, f"Done! Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
    if args.use_tensorboard:
        writer.add_text('VALID/Best', f"Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
        writer.close()

    # Final - Save best checkpoint as result model
    final_model_save_path = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type)
    final_model_save_name = f'final_model_{args.augmentation_type}.pt'

    check_path(final_model_save_path)
    shutil.copyfile(os.path.join(checkpoint_save_path, 'checkpoint.pt'), os.path.join(final_model_save_path, final_model_save_name)) # Copy best checkpoint as final model
    write_log(logger, f"FINAL - Saved final model to {final_model_save_path}")

    if args.use_wandb:
        wandb.finish()
