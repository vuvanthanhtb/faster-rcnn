import os
import torch
from pprintpp import pprint
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from voc_dataset import VOCDataset


def collate_fn(batch):
  images, labels = zip(*batch)
  return list(images), list(labels)


def get_args():
  parser = argparse.ArgumentParser(
      description='Train Faster R-CNN on VOC dataset')
  parser.add_argument('--year', '-y', type=str, default='2012',
                      help='Year of VOC dataset to use')
  parser.add_argument('--image_set', '-s', type=str, default='train',
                      help='Image set to use (train, val, trainval, test)')
  parser.add_argument('--data_path', '-d', type=str, default='data',
                      help='Path to VOC dataset root')
  parser.add_argument('--num_epochs', '-e', type=int, default=100,
                      help='Number of epochs to train for')
  parser.add_argument('--batch_size', '-b', type=int, default=4,
                      help='Batch size for training')
  parser.add_argument('--num_workers', '-w', type=int, default=4,
                      help='Number of workers for data loading')
  parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                      help='Learning rate for optimizer')
  parser.add_argument('--momentum', '-m', type=float, default=0.9,
                      help='Momentum for optimizer')
  parser.add_argument('--log_folder', '-l', type=str, default='tensorboard/pascal_voc',
                      help='Path to generated tensorboard')
  parser.add_argument('--checkpoint_path', '-c', type=str, default='checkpoints',
                      help='Path to save checkpoints')
  return parser.parse_args()


def train(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  transform = ToTensor()
  train_dataset = VOCDataset(
      root=args.data_path, year=args.year, image_set=args.image_set, transform=transform)

  train_dataloader = DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

  model = fasterrcnn_resnet50_fpn(
      weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
  in_channels = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(
      in_channels=in_channels, num_classes=len(train_dataset.categories))
  model.to(device)
  optimizer = torch.optim.SGD(params=model.parameters(
  ), lr=args.learning_rate, momentum=args.momentum)

  if not os.path.isdir(args.log_folder):
    os.makedirs(args.log_folder)

  writer = SummaryWriter(args.log_folder)
  num_iter_per_epoch = len(train_dataloader)

  for epoch in range(args.num_epochs):
    model.train()

    progress_bar = tqdm(train_dataloader, colour='cyan')
    for iter, (images, labels) in enumerate(progress_bar):
      images = [image.to(device) for image in images]
      labels = [{'boxes': target['boxes'].to(
          device), 'labels': target['labels'].to(device)} for target in labels]

      # Forward
      losses = model(images, labels)
      final_losses = sum([loss for loss in losses.values()])

      # Backward
      optimizer.zero_grad()
      final_losses.backward()
      optimizer.step()

      progress_bar.set_description(
          'Epoch {}/{}. Loss {:0.4f}'.format(epoch+1, args.num_epochs, final_losses.item()))
      writer.add_scalar('Loss/train', final_losses.item(),
                        epoch * num_iter_per_epoch + iter)


if __name__ == "__main__":
  args = get_args()
  train(args)
