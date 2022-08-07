import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import argparse
import wandb
from data.dataloader import LyftUdacity
from model.unet import unet_model
from albumentations.pytorch import ToTensorV2
import albumentations as A
from metrics.iou import iou_score
from model.trainer import train
torch.manual_seed(0)

def main(args):

    data_dir = [os.path.join(args.data_dir,'data'+i, 'data'+i) for i in ['A','B','C']]

    transform = A.Compose([
        A.Resize(160,240),
        A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406)
                                                , std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_batch,test_batch = get_images(args, data_dir,transform =transform,batch_size=args.batch_size,shuffle=True,pin_memory=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = unet_model().to(DEVICE)

    num_epochs = args.epochs

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(args.save_figure, exist_ok=True)

    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_fn = checkpoint['loss_fn']

        print("Loaded model from checkpoint: {}".format(args.model_path))
        train(args, model, num_epochs, loss_fn, optimizer, scaler, train_batch, test_batch, DEVICE)

    else:
        print("Training model from scratch (model not loaded)")
        train(args, model, num_epochs, loss_fn, optimizer, scaler, train_batch, test_batch, DEVICE)


def get_images(args, image_dir,transform=None,batch_size=1,shuffle=True,pin_memory=True):
    data = LyftUdacity(image_dir,transform = transform)
    train_size = int(0.8 * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=args.num_workers)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=args.num_workers)
    return train_batch,test_batch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./Semantic_segmentation_data/', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='set size of batch')
    parser.add_argument('--num_workers', type=int, default=8, help='set number of workers')
    parser.add_argument('--epochs', type=int, default=10, help='set number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='set learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='set momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='set weight decay')
    parser.add_argument('--save_model', type=bool, default=True, help='True if you want to save model')
    parser.add_argument('--save_figure', type=str, default='./test_image_save', help='path for saving test images')
    parser.add_argument('--load_model', type=bool, default=False, help='True if you want to load model')
    parser.add_argument('--model_path', type=str, default='./checkpoint/best_model.pth', help='path to pretrained model')
    parser.add_argument('--wandb', type=bool, default=False, help='True if you want to use wandb')
    parser.add_argument('--wandb_project', type=str, default='unet', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='YOUR_ID', help='wandb entitiy name')
    parser.add_argument('--wandb_score_warning_threshold', type=float, default=0.15, help='threshold for test score warning')


    args = parser.parse_args()

    print(args)
    
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_project, entity = args.wandb_entity) 
    else:
        print("wandb is not used")

    main(args)
