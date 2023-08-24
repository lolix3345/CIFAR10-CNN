from convnet import CNN_modulable_width
from dataset import Cifar10Dataset
from engine import SoftmaxEngine
import torch
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
import torchvision.transforms
import argparse

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='.\\data',
        help='Path to the directory containing the dataset files. The directory should contain the 5 pickle files containing the training images and labels, as well as the test set pickle file'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=1,
        help='Number of epoch you want to train the model'
    )
    parser.add_argument(
        '--printing-interval',
        type=int,
        default=30,
        help='Printing interval : the script will print the training information every --printing-interval batches'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Size of the batch for training and testing'
    )
    parser.add_argument(
        '--testing-interval',
        type=int,
        default=5,
        help='Will test the model every --testing-interval epochs on the test set during the training, and log/print the result'
    )
    parser.add_argument(
        '--model-width',
        type=int,
        default=16,
        help='Width of the CNN. The bigger, the more computationally expensive'
    )
    parser.add_argument(
        '--log-name',
        type=str,
        default='',
        help='name of the tensorboard log. If empty, there will be no logs'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Set this flag if you want to do training and inference with CUDA'
    )
    args = parser.parse_args()
    train_dataset = Cifar10Dataset(root_dir=args.dataset_dir, split='train')
    test_dataset  = Cifar10Dataset(root_dir=args.dataset_dir, split='test')
    model = CNN_modulable_width(width=args.model_width)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    logger = SummaryWriter("log/"+args.log_name) if args.log_name != '' else None
    engine = SoftmaxEngine(model=model,
                           dataset=train_dataset,
                           test_dataset=test_dataset,
                           optimizer=optimizer,
                           logger=logger,
                           batch_size=args.batch_size,
                           num_workers=0,
                           use_gpu=args.use_gpu)
    engine.train(num_epochs=args.n_epochs, eval_interval=args.testing_interval, print_interval=args.printing_interval)

if __name__ == '__main__':
    main()