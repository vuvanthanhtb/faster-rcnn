from pprintpp import pprint
import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor


class VOCDataset(VOCDetection):
  def __init__(self, root, year, image_set='train', download=False, transform=None, target_transform=None, transforms=None):
    super().__init__(root, year, image_set, download,
                     transform, target_transform, transforms)
    self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

  def __getitem__(self, index):
    image, data = super().__getitem__(index)
    all_bboxes = []
    all_labels = []
    for obj in data['annotation']['object']:
      xmin = int(obj['bndbox']['xmin'])
      ymin = int(obj['bndbox']['ymin'])
      xmax = int(obj['bndbox']['xmax'])
      ymax = int(obj['bndbox']['ymax'])
      all_bboxes.append([xmin, ymin, xmax, ymax])
      all_labels.append(self.categories.index(obj['name']))
    all_bboxes = torch.FloatTensor(all_bboxes)
    all_labels = torch.LongTensor(all_labels)
    target = {
        "boxes": all_bboxes,
        "labels": all_labels
    }
    return image, target


if __name__ == "__main__":
  transform = ToTensor()
  train_dataset = VOCDataset(
      root="data", year="2012", image_set="train", transform=transform)
  image, target = train_dataset[2000]
  pprint(target)
  pprint(image.shape)
  # image.show()
