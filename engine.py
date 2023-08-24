import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from misc import convert_seconds_to_hms
import numpy as np
import matplotlib.pyplot as plt

class SoftmaxEngine():
    def __init__(self,
                 model,
                 dataset,
                 test_dataset,
                 optimizer,
                 logger,
                 batch_size=32,
                 num_workers=0,
                 use_gpu=True):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = model
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logger
    def train(self, num_epochs=10, eval_interval=1, print_interval=10):
        self.model = self.model.to(self.device)
        self.model.train()
        train_start = time.time()
        batch_per_epoch = int(np.ceil(len(self.dataset)/self.batch_size))
        loss_sum = 0
        losses = []
        for current_epoch in range(num_epochs):
            epoch_start = time.time()
            current_batch = 0
            for (x, labels) in iter(self.dataloader):
                """Actual training step"""
                x, labels = x.to(self.device), labels.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                """End of actual training step"""
                """printing"""
                current_batch += 1
                loss_sum += loss.item()
                losses.append(loss.item())
                if current_batch % print_interval == 0:
                    current_time = time.time()
                    h_since_train_start, m_since_train_start, s_since_train_start = convert_seconds_to_hms(current_time - train_start)
                    batch_speed = (current_time - epoch_start)/current_batch
                    estimated_remaining_time = batch_speed*(batch_per_epoch - current_batch + (num_epochs - current_epoch - 1)*batch_per_epoch)
                    h_estimated, m_estimated, s_estimated = convert_seconds_to_hms(estimated_remaining_time)
                    print(f"{h_since_train_start:02}:{m_since_train_start:02}:{s_since_train_start:02} elapsed, epoch [{current_epoch+1}/{num_epochs}], batch [{current_batch}/{batch_per_epoch}], estimated {h_estimated:02}:{m_estimated:02}:{s_estimated:02} remaining, loss = {loss_sum/float(print_interval)}")
                    loss_sum = 0
                """logging in tensorboard"""
                if self.logger is not None:
                    self.logger.add_scalar("training loss", loss.item(), current_epoch*batch_per_epoch + current_batch)

            if (current_epoch + 1) % eval_interval == 0:
                print("computing test set accuracy...")
                test_accuracy = self.compute_test_accuracy()
                print(f"accuracy on test set : {(100*test_accuracy):.2f} %")
                if self.logger is not None:
                    self.logger.add_scalar("testing accuracy", test_accuracy, (current_epoch+1)*batch_per_epoch)

    def compute_test_accuracy(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        corrects = 0
        for x, labels in iter(self.test_dataloader):
            x, labels = x.to(self.device), labels.to(self.device)
            preds = self.model(x)
            preds = torch.argmax(preds, dim=1)
            equals = (preds == labels)
            corrects += torch.sum(equals).to("cpu").numpy()

        return float(corrects)/float(len(self.test_dataset))






