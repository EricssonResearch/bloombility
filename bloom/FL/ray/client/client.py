import ray
from torch import nn
import torch.optim as optim
import torch
import numpy as np


@ray.remote
class ParticipantActor:
    def __init__(self, model, iteration, train_data, test_data):
        self.model = model
        self.iteration = iteration
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.loss_f = nn.CrossEntropyLoss()
        self.losses = []
        model.train()

    def train(self, epochs):
        for epoch in range(epochs):
            loss_n = 0.0
            for features, labels in self.train_data:
                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.loss_f(predictions, labels)
                loss_n += loss.item()

                # backward propagation
                loss.backward()
                self.optimizer.step()
            self.losses.append(loss_n / len(self.train_data))
            print(
                f"Iteration {self.iteration}: Epoch {epoch+1} completed, loss: {np.average(self.losses)}"
            )
        return self.model

    def test(self):
        correct = 0.0
        total = 0.0
        total_loss = 0.0
        for features, labels in iter(self.test_data):
            predictions = self.model(features)
            _, predicted = predictions.max(1, keepdim=True)
            correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()
            total += len(predicted)
            total_loss += self.loss_f(predictions, labels).item()

        avg_loss = total_loss / len(self.test_data)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
