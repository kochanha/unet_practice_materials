# unet_practice_materials
2차 AI 실무 교육과정 5주차 실습자료

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/kochanha/unet_practice_materials.git
cd unet_practice_materials
pip install wandb
pip install -U git+https://github.com/albu/albumentations > /dev/null && echo
pip install torchmetrics
```

- Install [PyTorch](http://pytorch.org) and 1.10.1+ and other dependencies (e.g., torchvision).

### Training
- Download dataset with `data_download.sh`
```bash
bash data_download.sh
```

- Train a model:
```bash
usage: main.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--lr LR]
               [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
               [--save_model SAVE_MODEL] [--save_figure SAVE_FIGURE]
               [--load_model LOAD_MODEL] [--model_path MODEL_PATH]
               [--wandb WANDB] [--wandb_project WANDB_PROJECT]
               [--wandb_entity WANDB_ENTITY]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   path to dataset
  --batch_size BATCH_SIZE
                        set size of batch
  --num_workers NUM_WORKERS
                        set number of workers
  --epochs EPOCHS       set number of epochs
  --lr LR               set learning rate
  --momentum MOMENTUM   set momentum
  --weight_decay WEIGHT_DECAY
                        set weight decay
  --save_model SAVE_MODEL
                        True if you want to save model
  --save_figure SAVE_FIGURE
                        path for saving test images
  --load_model LOAD_MODEL
                        True if you want to load model
  --model_path MODEL_PATH
                        path to pretrained model
  --wandb WANDB         True if you want to use wandb
  --wandb_project WANDB_PROJECT
                        wandb project name
  --wandb_entity WANDB_ENTITY
                        wandb entitiy name

```
For example, run the following if not using Wandb
```
python main.py --data_dir './Semantic_segmentation_data/' --batch_size 16 --num_workers 8 --epochs 10 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --save_model True --save_figure './test_image_save'
```
To use Wandb,
```
python main.py --data_dir './Semantic_segmentation_data/' --batch_size 16 --num_workers 8 --epochs 10 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --save_model True --save_figure './test_image_save' --wandb True --wandb_project 'unet --wandb_entity 'PLACE_YOUR_ID'
```
