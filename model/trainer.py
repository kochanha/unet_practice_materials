import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model.unet import unet_model
from metrics.iou import iou_score
import wandb
import matplotlib.pyplot as plt
from test import print_iou
from PIL import Image

def train(args, model, num_epochs, loss_fn, optimizer, scaler, train_batch, test_batch, DEVICE):

    print("-----Training for {} epochs-----".format(num_epochs))
    
    test_score_threshold = args.wandb_score_warning_threshold
    
    for epoch in range(num_epochs):
        print("epoch : {} / {}".format(epoch+1, num_epochs))
        loop = tqdm(enumerate(train_batch),total=len(train_batch))

        for batch_idx, (data, targets) in loop:

            model.train()

            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx %50 ==0:
                model.eval()
                with torch.no_grad():
                    preds = torch.argmax(predictions,axis=1).to('cpu')
                    mask1 = np.array(preds[0,:,:])
                    img = plt.imshow(mask1)
                    plt.savefig('./{}/{}_{}.png'.format(args.save_figure,epoch+1,batch_idx))

                    if args.wandb:
                        wandb.log({'predicted map' : [wandb.Image(img)]})
                        
            loop.set_postfix(loss=loss.item())
            if args.wandb:
                wandb.log({'Epoch': epoch+1, 'Loss': loss })

        model.eval()
        test_score = print_iou(test_batch, model)
        if args.wandb:
            if test_score_threshold > test_score:
                wandb.alert(
                    title='Low Test Score',
                    text=f'IoU score {test_score} at epoch {epoch+1} is below the acceptable theshold, {test_score_threshold}',
                )
                print('Low Score Alert triggered')
            wandb.log({'test iou': test_score })
        
        if args.save_model:
            checkpoint_path = './checkpoint'
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn' : loss_fn
                }, "{}/epoch_{}_loss_{:.2f}_testiou_{:.2f}.pth".format(checkpoint_path, epoch+1, loss, test_score))

            print("Saved checkpoint, {}/epoch_{}_loss_{:.2f}_testiou_{:.2f}.pth \n".format(checkpoint_path, epoch, loss, test_score))
        
        else:
            print("model not saved")
    
    print("Training finished")
    model.eval()
    print_iou(test_batch, model)
