import random

import torchvision.transforms as T


class RandomRotate90:
  """Randomly rotate by 0/90/180/270 degrees for orientation-robust remote sensing views."""

  def __call__(self, image):
    angle = 90 * random.randint(0, 3)
    if angle == 0:
      return image
    return image.rotate(angle)


def build_weak_transform(image_size: int):
  """Weak augmentation used by DKD-style unlabeled branch."""
  return T.Compose([
    T.RandomResizedCrop(image_size),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    RandomRotate90(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])


def build_strong_transform(image_size: int):
  """Strong augmentation used by DKD-style unlabeled branch."""
  return T.Compose([
    T.RandomResizedCrop(image_size),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    RandomRotate90(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])


__all__ = ["build_weak_transform", "build_strong_transform"]
