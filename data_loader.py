
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

#download data
def creat_dataloader():
  ROOT = './data'
  train_data = datasets.MNIST(
      root=ROOT,
      download = True,
      train= True
  )
  test_data = datasets.MNIST(
      root=ROOT,
      train = False,
      download = True
  )

  #split training

  valid_ratio = 0.9

  n_train_examples = int(len(train_data)*valid_ratio)
  n_valid_example = len(train_data)-n_train_examples

  train_data,valid_data = data.random_split(
      train_data,
      [n_train_examples,n_valid_example]
  )
  num_classes = len(train_data.dataset.classes)
  #compute mean anh std for normalization

  mean = train_data.dataset.data.float().mean()/255
  std = train_data.dataset.data.float().std()/255

  train_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[mean],std=[std])
  ])

  test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[mean],std=[std])
  ])
  train_data.dataset.transform = train_transforms
  valid_data.dataset.transform = test_transforms

  #Create dataloader
  Batch_size = 256

  train_dataloader = data.DataLoader(
      train_data,
      shuffle = True,
      batch_size = Batch_size
  )
  valid_dataloader = data.DataLoader(
      valid_data,
      batch_size = Batch_size
  )
  return train_dataloader,valid_dataloader,num_classes
