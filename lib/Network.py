import torch.nn as nn
import torch
import time
import numpy as np
from torch import optim
from collections import defaultdict


def update_classwise_accuracies(preds, labels, class_correct, class_totals):
    correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
    for i in range(labels.shape[0]):
        label = labels.data[i].item()
        class_correct[label] += correct[i].item()
        class_totals[label] += 1


def get_accuracies(class_names, class_correct, class_totals):
    accuracy = (100 * np.sum(list(class_correct.values())) / np.sum(list(class_totals.values())))
    class_accuracies = [(class_names[i], 100.0 * (class_correct[i] / class_totals[i]))
                        for i in class_names.keys() if class_totals[i] > 0]
    return accuracy, class_accuracies


class Network(nn.Module):

    def __init__(self, device=None):
        super().__init__()

        self.best_accuracy = 0.

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def forward(self, *input):
        pass

    def fit(self, trainloader, validloader, epochs=2, print_every=10, validate_every=1):
        for epoch in range(epochs):
            self.model.to(self.device)
            print('epoch {:3d}/{}'.format(epoch + 1, epochs))
            epoch_train_loss = self.train_(trainloader, self.criterion,
                                           self.optimizer, print_every)

            if validate_every and (epoch % validate_every == 0):
                t2 = time.time()
                epoch_validation_loss, epoch_accuracy = self.validate_(validloader)
                time_elapsed = time.time() - t2
                print(f"{time.asctime()}--Validation time {time_elapsed:.3f} seconds.."
                      f"Epoch {epoch+1}/{epochs}.. "
                      f"Epoch Training loss: {epoch_train_loss:.3f}.. "
                      f"Epoch validation loss: {epoch_validation_loss:.3f}.. "
                      f"validation accuracy: {epoch_accuracy:.3f}")

                if self.best_accuracy == 0. or (epoch_accuracy > self.best_accuracy):
                    print('updating best accuracy: previous best = '
                          '{:.3f} new best = {:.3f}'.format(self.best_accuracy, epoch_accuracy))

                    self.best_accuracy = epoch_accuracy

                torch.save(self.state_dict(), self.best_accuracy_file)

            self.train()
        print('loading best accuracy model')
        self.load_state_dict(torch.load(self.best_accuracy_file))

    def set_criterion(self, criterion_name):
        if criterion_name.lower() == 'nllloss':
            self.criterion_name = 'NLLLoss'
            self.criterion = nn.NLLLoss()
        elif criterion_name.lower() == 'crossentropyloss':
            self.criterion_name = 'CrossEntropyLoss'
            self.criterion = nn.CrossEntropyLoss()

    def set_optimizer(self, params, optimizer_name='adam', lr=0.003):
        if optimizer_name.lower() == 'adam':
            print('setting optim Adam')
            self.optimizer = optim.Adam(params, lr=lr)
            self.optimizer_name = optimizer_name

        elif optimizer_name.lower() == 'sgd':
            print('setting optim SGD')
            self.optimizer = optim.SGD(params, lr=lr)
        elif optimizer_name.lower() == 'adadelta':
            print('setting optim Ada Delta')
            self.optimizer = optim.Adadelta(params)

    def set_model_params(self,
                         criterion_name,
                         optimizer_name,
                         lr,  # learning rate
                         dropout_p,  # dropout probabilities
                         model_name,
                         best_accuracy,
                         best_accuracy_file,
                         class_names):
        self.set_criterion(criterion_name)
        self.set_optimizer(self.parameters(), optimizer_name, lr=lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name = model_name
        self.best_accuracy = best_accuracy
        self.best_accuracy_file = best_accuracy_file
        self.class_names = class_names

    def get_model_params(self):
        params = {
            'device': self.device,
            'model_name': self.model_name,
            'optimizer_name': self.optimizer_name,
            'criterion_name': self.criterion_name,
            'lr': self.lr,
            'dropout_p': self.dropout_p,
            'best_accuracy': self.best_accuracy,
            'best_accuracy_file': self.best_accuracy_file,
            'class_names': self.class_names}

        return params

    def set_model_params(self,
                         criterion_name,
                         optimizer_name,
                         lr,  # learning rate
                         dropout_p,
                         model_name,
                         best_accuracy,
                         best_accuracy_file,
                         chkpoint_file):
        self.criterion_name = criterion_name
        self.set_criterion(criterion_name)
        self.optimizer_name = optimizer_name
        self.set_optimizer(self.parameters(), optimizer_name, lr=lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name = model_name
        self.best_accuracy = best_accuracy
        print('set_model_params: best accuracy = {:.3f}'.format(self.best_accuracy))
        self.best_accuracy_file = best_accuracy_file
        self.chkpoint_file = chkpoint_file

    def get_model_params(self):
        params = {}

        params['device'] = self.device
        params['model_name'] = self.model_name
        params['optimizer_name'] = self.optimizer_name
        params['criterion_name'] = self.criterion_name
        params['lr'] = self.lr
        params['dropout_p'] = self.dropout_p
        params['best_accuracy'] = self.best_accuracy
        print('get_model_params: best accuracy = {:.3f}'.format(self.best_accuracy))
        params['best_accuracy_file'] = self.best_accuracy_file
        params['chkpoint_file'] = self.chkpoint_file
        print('get_model_params: chkpoint file = {}'.format(self.chkpoint_file))
        return params

    def save_chkpoint(self):
        saved_model = {}
        saved_model['params'] = self.get_model_params()
        torch.save(saved_model, self.chkpoint_file)
        print('checkpoint created successfully in {}'.format(self.chkpoint_file))

    def train_(self, trainloader, criterion, optimizer, print_every):
        self.train()
        t0 = time.time()
        batches = 0
        running_loss = 0

        for inputs, labels in trainloader:
            batches += 1
            # t1 = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            running_loss += loss

            if batches % print_every == 0:
                print(f"{time.asctime()}.."
                      f"Time Elapsed = {time.time()-t0:.3f}.."
                      f"Batch {batches+1}/{len(trainloader)}.. "
                      f"Average Training loss: {running_loss/(batches):.3f}.. "
                      f"Batch Training loss: {loss:.3f}.. ")
            t0 = time.time()
            return running_loss / len(trainloader)

    def validate_(self, validloader, every=1):
        running_loss = 0
        class_correct = defaultdict(int)
        class_totals = defaultdict(int)
        show = 1

        self.eval()

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(torch.exp(outputs), 1)
                if show >= every:
                    update_classwise_accuracies(preds, labels, class_correct, class_totals)
                    show = 0

                show += 1

        accuracy = (100 * np.sum(list(class_correct.values())) / np.sum(list(class_totals.values())))
        self.train()
        return running_loss / len(validloader), accuracy

    def evaluate(self, testloader, every=1):
        self.eval()
        show = 1

        self.model.to(self.device)
        class_correct = defaultdict(int)
        class_totals = defaultdict(int)
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                ps = torch.exp(outputs)
                _, preds = torch.max(ps, 1)
                if show >= every:
                    update_classwise_accuracies(preds, labels, class_correct, class_totals)
                    show = 0

                show += 1

        self.train()
        return get_accuracies(self.class_names, class_correct, class_totals)

    def predict(self, inputs, topk=1):
        self.eval()

        self.model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
            ps = torch.exp(outputs)
            p, top = ps.topk(topk, dim=1)
        return p, top
