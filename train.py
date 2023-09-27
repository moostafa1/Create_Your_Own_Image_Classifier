import os
import json
import argparse
import numpy as np
from os import listdir # , getcwd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
# import torch.nn.functional as F



def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments using add_argument() from ArguementParser method
    parser.add_argument('--dir', type = str, default = 'flowers/', help = 'path to the folder of images')
    parser.add_argument('--save_dir', type = str, default = 'saved_models/', help = 'place to save checkpoints and files')
    parser.add_argument('--arch', type = str, default = 'vgg19', help = 'used pre-trained model')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'learning_rate')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'number of hidden units')
    parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs')
    parser.add_argument('--gpu', type = str, default = 'cuda:0', help = 'device use to train the model on')

    return parser.parse_args()



def check_command_line_arguments(in_arg):
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print(in_arg)



class Classifier:
    def __init__(self, data_dir='flowers', model_name='densenet121', input_size=1024, hidden_layers=512,
                 output=102, loss_function_name='NLLLoss', optimizer_name='adam', epochs=15, dropout=.2,
                 learnrate=.002, save_dir='saved_models/'):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output = output
        self.epochs = epochs
        self.data_dir = data_dir
        self.save_dir = save_dir

        models_dict = {'vgg13': models.vgg13(pretrained=True), 'vgg16': models.vgg16(pretrained=True),
                       'vgg19': models.vgg19(pretrained=True), 'alexnet': models.alexnet(pretrained=True),
                       'resnet18': models.resnet18(pretrained=True), 'densenet121': models.densenet121(pretrained=True)}

        classifier = nn.Sequential(nn.Linear(input_size, hidden_layers),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_layers, output),
                                   nn.LogSoftmax(dim=1))

        if model_name in models_dict:
            model = models_dict[model_name]
        elif model_name.split('_')[0] == 'old':
            #suggested = [pt for pt in listdir() if (pt.endswith('.pt') or pt.endswith('.pth')) and pt.split('_')[0] == model_name.split('_')[1]]
            suggested = [pt for pt in listdir() if pt.endswith('.pt') or pt.endswith('.pth')]
            print(suggested)
            ind = input("select by index which model to load: ")
            checkpoint = torch.load(suggested[ind])
            model = models_dict[model_name.split('_')[1]]
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f'model name not exist.\nPlease try: {list(models_dict)} or "old_model_name" if you have a model')
            exit(0)

        # freeze model parameters to avoid backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = classifier
        model.class_to_idx = datasets.ImageFolder(self.data_dir + '/train').class_to_idx
        self.model = model

        loss_funcs = {'cross_entropy': nn.CrossEntropyLoss(), 'NLLLoss': nn.NLLLoss()}
        self.criterion = loss_funcs[loss_function_name]

        optimizers = {'adam': optim.Adam(model.classifier.parameters(), lr=learnrate),
                      'sgd': optim.SGD(model.classifier.parameters(), lr=learnrate)}
        self.optimizer = optimizers[optimizer_name]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device);


    def dataloader(self, train_transforms, test_transforms, batch_size=64):
        # Pass transforms in here, then run the next cell to see how the transforms look
        train_data = datasets.ImageFolder(self.data_dir + '/train', transform=train_transforms)
        test_data = datasets.ImageFolder(self.data_dir + '/test', transform=test_transforms)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        try:
            valid_data = datasets.ImageFolder(self.data_dir + '/valid', transform=train_transforms)
            validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
            return trainloader, validloader, testloader
        except:
            return trainloader, testloader

    def save_loss_vals(self, train_losses, test_losses, fname='train_test_losses.json'):
        loss_dict = {'train_losses': train_losses, 'test_losses': test_losses}
        with open(self.save_dir + fname, 'w') as f:
            json.dump(loss_dict, f)

    def load_json_data(self, fname):
        with open(self.save_dir + fname, 'r') as f:
            data_dict = json.load(f)
        return data_dict #['train_losses'], loss_dict['test_losses']

    def plot_loss_vals(self, train_loss, valid_loss):
        plt.plot(train_loss, label='Training loss')
        plt.plot(valid_loss, label='Testing loss')
        plt.legend(frameon=False);

    def save_model(self, fname='checkpoint.pth'):
        state = {
        'input_size': self.input_size,
        'hidden_size': self.hidden_layers,
        'output_size': self.output,
        'model_state_dict': self.model.state_dict(),
        'class_to_idx': self.model.class_to_idx,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epochs': self.epochs
        }
        torch.save(state, self.save_dir + fname)
        print(f"model saved to {fname}")

    def load_model(self, pth_fname='checkpoint.pth'):
        checkpoint = torch.load(self.save_dir + pth_fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        output_size = checkpoint['output_size']
        self.model.class_to_idx: checkpoint['class_to_idx']
        epochs = checkpoint['epochs']
        return epochs, input_size, hidden_size, output_size





class TrainModel(Classifier):
    def __init__(self, data_dir='flowers', model_name='densenet121', input_size=1024,
                 hidden_layers=512, output=102, loss_function_name='NLLLoss',optimizer_name='adam',
                 epochs=15, dropout=.2,learnrate=.002, save_dir='saved_models/'):
        super().__init__(data_dir, model_name, input_size, hidden_layers, output,
                         loss_function_name, optimizer_name, epochs, dropout, learnrate, save_dir)

    def train(self, trainloader, validloader, epochs=7, stop_threash=-2,
              print_every=60, save_checkpoint=True, save_loss=True, plot_loss=True):
        steps = 0
        train_loss = 0
        train_losses, valid_losses = [], []
        for e in range(epochs):
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()

                log_ps = self.model(images)
                loss = self.criterion(log_ps, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_losses.append(train_loss)

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for images, labels in validloader:
                            images, labels = images.to(device), labels.to(device)
                            log_ps = self.model(images)
                            loss = self.criterion(log_ps, labels)

                            valid_loss += loss.item()
                            valid_losses.append(valid_loss)

                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch: {e+1}/{epochs}.. "
                          f"Train loss: {train_loss/len(trainloader):.3f}.. "
                          f"Test loss: {valid_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validloader):.3f}")
                    train_loss = 0
                    self.model.train()
                    # for early stopping
                    if (train_loss/len(trainloader) - valid_loss/len(validloader) < stop_threash):
                        break
        if save_checkpoint:
            self.save_model()
        if save_loss:
            self.save_loss_vals(train_losses, valid_losses)
        if plot_loss:
            self.plot_loss_vals(train_losses, valid_losses)


    def test(self, testloader):
        with torch.no_grad():
            test_loss = 0
            accuracy = 0
            self.model.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = self.model(images)
                loss = self.criterion(log_ps, labels)

                test_loss += loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}")




class DataProcessing:
    def __init__(self):
        pass

    def transform(self, rotation=30, crop=224, resize=255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # TODO: Define transforms for the training data and testing data
        train_transforms = transforms.Compose([transforms.RandomRotation(rotation),
                                               transforms.RandomResizedCrop(crop),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)])

        test_transforms = transforms.Compose([transforms.Resize(resize),
                                              transforms.CenterCrop(crop),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        return train_transforms, test_transforms


    def imshow(self, image, ax=None, title=None, normalize=True):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        image = image.numpy().transpose((1, 2, 0))

        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

        plt.title(title)
        ax.imshow(image)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        return ax





if __name__ == "__main__":
    # receive command line arguments from user
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg
    check_command_line_arguments(in_arg)


    # define the transformations
    train_transforms, test_transforms = DataProcessing().transform()

    # define dataloaders
    trainloader, validloader, testloader = Classifier(in_arg.dir, in_arg.arch, hidden_layers=in_arg.hidden_units, epochs=in_arg.epochs, learnrate=in_arg.learning_rate, save_dir=in_arg.save_dir).dataloader(train_transforms, test_transforms)

    # set the device to 'GPU' if possible
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = in_arg.gpu
    print(f"You are training your model on --- {device} ---\n\n")

    # creates the classifier
    classifier = TrainModel(in_arg.dir, in_arg.arch, hidden_layers=in_arg.hidden_units, epochs=in_arg.epochs, learnrate=in_arg.learning_rate, save_dir=in_arg.save_dir)
    print(f"Your model name is: {classifier.model.__class__.__name__}")

    # train the classifier
    print("Training has been started:")
    classifier.train(trainloader, validloader, in_arg.epochs)

    # Test the model accuracy on test dataset
    print("Testing has been started:")
    classifier.test(testloader)