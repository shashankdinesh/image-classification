import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


transformations = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_total_dataset = datasets.ImageFolder("/content/drive/My Drive/IMAGE_RECOGNITION/TRAIN", transform = transformations)
train_dataset_loader = DataLoader(dataset = train_total_dataset, batch_size = 100)

val_total_dataset = datasets.ImageFolder("/content/drive/My Drive/IMAGE_RECOGNITION/VAL", transform = transformations)
val_dataset_loader = DataLoader(dataset = val_total_dataset, batch_size = 10)

test_total_dataset = datasets.ImageFolder("/content/drive/My Drive/IMAGE_RECOGNITION/VAL", transform = transformations)
test_dataset_loader = DataLoader(dataset = test_total_dataset, batch_size = 10)