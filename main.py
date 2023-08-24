from convnet import CNN, CNN_modulable_width
from transformers import ImageTransformer
from dataset import Cifar10Dataset
from engine import SoftmaxEngine
import torch
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
import torchvision.transforms
transforms =torchvision.transforms.Compose(
    [torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation([0,90]),
     torchvision.transforms.ColorJitter()]
)
dataset = Cifar10Dataset(root_dir="D:\\DeepLearnings\\dataset\\cifar-10-python\\cifar-10-batches-py", transform=transforms,split='train')
test_dataset = Cifar10Dataset(root_dir="D:\\DeepLearnings\\dataset\\cifar-10-python\\cifar-10-batches-py", split='test')
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
logger = SummaryWriter("log/ImageTransformer_4x4_patches_with_transforms_epoch_300")
engine = SoftmaxEngine(model=model,
                       dataset=dataset,
                       test_dataset=test_dataset,
                       optimizer=optimizer,
                       logger=logger,
                       batch_size=128,
                       num_workers=0)
engine.train(num_epochs=300,eval_interval=5, print_interval=30)