# Standard Library Modules
import os
import argparse
# Custom Modules
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.user_name = os.getlogin()
        self.proj_name = 'Colorful_Cutout_Aug'

        # Task arguments
        task_list = ['classification']
        self.parser.add_argument('--task', type=str, choices=task_list, default='classification',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'tiny_imagenet']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='mnist',
                                 help='Dataset for the task; Must be given.')
        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')

        # Path arguments - Modify these paths to fit your environment
        self.parser.add_argument('--data_path', type=str, default='./dataset/',
                                 help='Path to the raw dataset before preprocessing.')
        self.parser.add_argument('--preprocess_path', type=str, default=f'/HDD/{self.user_name}/preprocessed/{self.proj_name}',
                                 help='Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'/HDD/{self.user_name}/model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'/HDD/{self.user_name}/model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type=str, default=f'/HDD/{self.user_name}/results/{self.proj_name}',
                                 help='Path to the result after testing.')
        self.parser.add_argument('--log_path', type=str, default=f'/HDD/{self.user_name}/tensorboard_log/{self.proj_name}',
                                 help='Path to the tensorboard log file.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default=self.proj_name,
                                 help='Name of the project.')
        model_type_list = ['vgg11', 'resnet50', 'resnet152', 'efficientnet_b0', 'efficientnet_b7', 'vit_b_16']
        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='resnet50',
                                 help='Type of the classification model to use.')
        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--dropout_rate', type=float, default=0.2,
                                 help='Dropout rate of the model; Default is 0.2')
        self.parser.add_argument('--image_resize_size', type=int, default=256,
                                 help='Image resize size; Default is 256')
        self.parser.add_argument('--image_crop_size', type=int, default=224,
                                 help='Image crop size; Default is 224')

        # Model - Size arguments
        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='Adam',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='None',
                                 help="Scheduler to use for classification; If None, no scheduler is used; Default is None")

        # Training arguments 1
        self.parser.add_argument('--num_epochs', type=int, default=50,
                                 help='Training epochs; Default is 50')
        self.parser.add_argument('--learning_rate', type=float, default=5e-5,
                                 help='Learning rate of optimizer; Default is 5e-5')
        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch size; Default is 32')
        self.parser.add_argument('--weight_decay', type=float, default=0,
                                 help='Weight decay; Default is 5e-4; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=5,
                                 help='Gradient clipping norm; Default is 5')
        self.parser.add_argument('--label_smoothing_eps', type=float, default=0.05,
                                 help='Label smoothing epsilon; Default is 0.05')
        self.parser.add_argument('--early_stopping_patience', type=int, default=5,
                                 help='Early stopping patience; No early stopping if None; Default is 5')
        self.parser.add_argument('--train_valid_split', type=float, default=0.1,
                                 help='Train/Valid split ratio; Default is 0.1')
        objective_list = ['loss', 'accuracy', 'f1']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='accuracy',
                                 help='Objective to optimize; Default is accuracy')

        # Augmentation arguments
        augmentation_type_list = ['none', 'cutout', 'color_cutout_cur', 'color_cutout_nocur', 'mixup', 'cutmix']
        self.parser.add_argument('--augmentation_type', type=str, choices=augmentation_type_list, default='none',
                                 help='Whether to use augmented data; Default is none')
        self.parser.add_argument('--augmentation_box_size', type=int, default=32,
                                 help='Augmentation box size; Default is 32')
        self.parser.add_argument('--augmentation_mixup_alpha', type=float, default=0.2,
                                 help='Mixup alpha; Default is 0.2')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=16, type=int,
                                 help='Batch size for test; Default is 16')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=None,
                                 help='Random seed; Default is None')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=True,
                                 help='Using tensorboard; Default is True')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb; Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()