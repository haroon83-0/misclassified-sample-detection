import os
import time
import ssl
import click
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from joblib import dump, load
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from prettytable import PrettyTable

# Global Variables
# Used to record the current convolutional layer index during traversal
th = 0
# Number of paths to extract
path_count_to_extract = 0
# Directory to save the extracted paths
global_path_save_dir = './'
# Directory to save the extracted features
global_features_save_dir = './'
# Prediction labels for training data
global_pred_list = None
# True labels for training data
global_label_list = None
# To store intermediate layer results
layers = []
# Random permutations
perms = []
# Training dataset
trainloader = None
# Backup of the training dataset
trainloader_copy = None
# Testing dataset
testloader = None
# Backup of the testing dataset
testloader_copy = None
# Custom dataset
dataloader = None
# Backup of the custom dataset
dataloader_copy = None
# Custom dataset names
data_name_list = None

# Use GPU for training, this can be set in the menu "Runtime" -> "Change Runtime Type"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ssl._create_default_https_context = ssl._create_unverified_context

buffer = ["─", "\\", "|", "/"]

# Class labels for different datasets
classes = None

provider_classes = None

classes_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

classes_fashion_mnist = ['T-shirt', 'Pants', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

classes_cifar10 = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def loadData(data_dir, data_type, download):
    global trainloader
    global trainloader_copy
    global testloader
    global testloader_copy
    global classes

    if data_type == 'MNIST':
        classes = classes_mnist
        transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        transform_copy = transforms.Compose([
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=download,
                                              transform=transform)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=download,
                                             transform=transform)
        trainset_copy = torchvision.datasets.MNIST(root=data_dir, train=True, download=download,
                                                   transform=transform_copy)
        testset_copy = torchvision.datasets.MNIST(root=data_dir, train=False, download=download,
                                                  transform=transform_copy)

    elif data_type == 'FASHION_MNIST':
        classes = classes_fashion_mnist
        transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        transform_copy = transforms.Compose([
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True,
                                                     download=download,
                                                     transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False,
                                                    download=download,
                                                    transform=transform)
        trainset_copy = torchvision.datasets.FashionMNIST(root=data_dir, train=True,
                                                          download=download,
                                                          transform=transform_copy)
        testset_copy = torchvision.datasets.FashionMNIST(root=data_dir, train=False,
                                                         download=download,
                                                         transform=transform_copy)
    elif data_type == 'CIFAR10':
        classes = classes_cifar10
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download,
                                               transform=transform_test)
        trainset_copy = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download,
                                                     transform=transforms.Compose([transforms.ToTensor()]))
        testset_copy = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download,
                                                    transform=transforms.Compose([transforms.ToTensor()]))

    else:
        print("\033[0;31;40m", 'Unsupported data type!!!', "\033[0m")
        exit(1)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    trainloader_copy = {k: v for (k, v) in
                        enumerate(torch.utils.data.DataLoader(trainset_copy, batch_size=1, shuffle=False))}
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    testloader_copy = {k: v for (k, v) in
                       enumerate(torch.utils.data.DataLoader(testset_copy, batch_size=1, shuffle=False))}


def loadDataProvider(data_dir, data_type):
    global dataloader
    global dataloader_copy
    global data_name_list
    global provider_classes
    global classes

    if data_type == 'MNIST':
        classes = classes_mnist
        transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        transform_copy = transforms.Compose([
            transforms.ToTensor()
        ])
    elif data_type == 'FASHION_MNIST':
        classes = classes_fashion_mnist
        transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        transform_copy = transforms.Compose([
            transforms.ToTensor()
        ])
    elif data_type == 'CIFAR10':
        classes = classes_cifar10
        transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_copy = transforms.Compose([
            transforms.ToTensor()])
    else:
        print("\033[0;31;40m", 'Unsupported data type!!!', "\033[0m")
        exit(1)
    try:
        data_provider = torchvision.datasets.ImageFolder(data_dir, transform=transform)
        data_provider_copy = torchvision.datasets.ImageFolder(data_dir, transform=transform_copy)
        provider_classes = data_provider.classes
        data_name_list = list(map(lambda item: os.path.split(item[0])[-1], data_provider.imgs))
        dataloader = torch.utils.data.DataLoader(data_provider, batch_size=1, shuffle=False)
        dataloader_copy = {k: v for (k, v) in
                           enumerate(torch.utils.data.DataLoader(data_provider_copy, batch_size=1, shuffle=False))}
    except BaseException as e:
        print("\033[0;31;40m", e, "\033[0m")
        exit(1)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(self.cfg)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.Conv2d(64, 192, kernel_size=4, stride=1, padding=1),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.AdaptiveAvgPool2d(output_size=(3, 3))
        )
        self.classifier = nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(256 * 3 * 3, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(in_features=16 * 4 * 4, out_features=10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 4 * 4)
        x = self.classifier(x)
        return x


@click.group()
def cli():
    pass


@click.command()
@click.option('--data_dir', prompt='input data directory to download',
              help='the directory of dataset to download')
@click.option('--data_type', prompt='input data type [MNIST/FASHION_MNIST/CIFAR10]',
              help='the type of dataset [MNIST/FASHION_MNIST/CIFAR10]')
def download(data_dir, data_type):
    try:
        loadData(data_dir, data_type, True)
    except BaseException as e:
        print("\033[0;31;40m", e, "\033[0m")
        exit(1)


@click.command(name='path-extract')
@click.option('--model_dir', prompt='input model directory', help='the directory of MUT')
@click.option('--path_save_dir', prompt='input path directory to save', default='./',
              help='the directory of path to save')
@click.option('--count', prompt='input path count to extract',
              help='the number of path to extract', default=18)


def path_extract(model_dir, path_save_dir, count):
    global path_count_to_extract
    path_count_to_extract = count
    global global_path_save_dir
    global_path_save_dir = path_save_dir
    # Load the model
    net = torch.load(model_dir, map_location=torch.device('cpu'))

    # Set to evaluation mode
    net.eval()

    # Generate random permutations for each convolutional layer
    net.apply(generate_perms)

    # Save the extracted path information
    save_path()

    # Print success message in green
    print("\033[1;32m", 'Successfully!!!', "\033[0m")


@click.command(name='features-extract')
@click.option('--model_dir', prompt='input model directory', help='the directory of MUT')
@click.option('--path_load_dir', prompt='input path directory to load', default='./',
              help='the directory of path to load')
@click.option('--features_save_dir', prompt='input features directory to save', default='./',
              help='the directory of features to save')
@click.option('--data_dir', prompt='input data directory to exec',
              help='the directory of dataset to exec')
@click.option('--data_type', prompt='input data type [MNIST/FASHION_MNIST/CIFAR10]',
              help='the type of dataset [MNIST/FASHION_MNIST/CIFAR10]')


def features_extract(model_dir, path_load_dir, features_save_dir, data_dir, data_type):
    global global_features_save_dir
    global_features_save_dir = features_save_dir

    # Load the model
    net = torch.load(model_dir, map_location=torch.device('cpu'))

    # Set to evaluation mode
    net.eval()

    try:
        # Load the data
        loadData(data_dir, data_type, False)
    except BaseException as e:
        print("\033[0;31;40m", e, "\033[0m")
        exit(1)

    # Generate feature list for each convolutional layer
    net.apply(generate_layer)

    # Register hook functions
    net.apply(register_hook)

    # Load the extracted path information
    load_path(path_load_dir)

    # Prediction labels
    pred_list = []
    # True labels
    label_list = []

    # Extract features
    with tqdm(total=len(trainloader), desc='Extracting features, please wait', leave=True, unit='B', unit_scale=True) as pbar:
        with torch.no_grad():
            for i, data in enumerate(trainloader):
                image, label = data
                image, label = image.to(device), label.to(device)
                outputs = net(image)
                _, predicted = torch.max(outputs.data, 1)
                label_list.append(label)
                pred_list.append(predicted)
                pbar.update(1)

    # Save the extracted features
    save_features(pred_list, label_list)
    print("\033[1;32m", 'successfully!!!', "\033[0m")


@click.command(name='train-binary-classifier')
@click.option('--model_dir', prompt='input model directory', help='the directory of MUT')
@click.option('--path_load_dir', prompt='input path directory to load', default='./',
              help='the directory of path to load')
@click.option('--features_load_dir', prompt='input features directory to load', default='./',
              help='the directory of features to load')
@click.option('--classifier_dir', prompt='input classifier directory to save', default='./',
              help='the directory of classifier to save')
@click.option('--classifier_type', prompt='input classifier type [SVM/RF/LR/DT/AB/GNB]', default='SVM',
              help='the type of classifier [SVM/RF/LR/DT/AB/GNB]')


def train_binary_classifier(model_dir, path_load_dir, features_load_dir, classifier_dir, classifier_type):
    # Load the model
    net = torch.load(model_dir, map_location=torch.device('cpu'))

    # Set to evaluation mode
    net.eval()

    # Load path information
    load_path(path_load_dir)

    # Load the features to be used for training
    load_features(features_load_dir)

    # Build the feature dataset for training the binary classification model
    all_features_to_train = []
    with tqdm(total=len(layers[0]), desc='Constructing training data, please wait', leave=True, unit='B', unit_scale=True) as pbar:
        for idx in range(len(layers[0])):
            # Aggregate features for the idx-th sample of each layer
            ll = []
            for layer_th in range(len(layers)):
                ll.append(layers[layer_th][idx])
            res = torch.stack(tuple(ll))
            res = res.t()
            all_features_to_train.append(res)
            pbar.update(1)

    # Complete feature set for all samples to be trained
    all_features_to_train = torch.stack(all_features_to_train)

    # Train the classifier
    to_train(classifier_dir, classifier_type, all_features_to_train)

    print("\033[1;32m", 'Train successfully!!!', "\033[0m")



@click.command(name='execute-result')
@click.option('--model_dir', prompt='input model directory', help='the directory of MUT')
@click.option('--path_load_dir', prompt='input path directory to load', default='./',
              help='the directory of path to load')
@click.option('--classifier_load_dir', prompt='input classifier directory to load', default='./',
              help='the directory of classifier to load')
@click.option('--data_dir', prompt='input data directory to exec',
              help='the directory of dataset to exec')
@click.option('--data_type', prompt='input data type [MNIST/FASHION_MNIST/CIFAR10]',
              help='the type of dataset [MNIST/FASHION_MNIST/CIFAR10]')
@click.option('--is_test_set', prompt='input whether to execute the test set [yes/no]', default='yes',
              help='input whether to execute the test set')


def execute_result(model_dir, path_load_dir, classifier_load_dir, data_dir, data_type, is_test_set):
    # Load the model
    net = torch.load(model_dir, map_location=torch.device('cpu'))

    # Set to evaluation mode
    net.eval()

    # Generate feature list for each convolutional layer
    net.apply(generate_layer)

    # Register hook functions
    net.apply(register_hook)

    try:
        # Load the data
        loadData(data_dir, data_type, False)
    except BaseException as e:
        print("\033[0;31;40m", e, "\033[0m")
        exit(1)

    # Load extracted path information
    load_path(path_load_dir)

    # Load the classifier
    clf = load_classifier(classifier_load_dir)

    # Execute results
    execute(net, clf, is_test_set)


@click.command(name='start-detect')
@click.option('--model_dir', prompt='input model directory', help='the directory of MUT')
@click.option('--path_load_dir', prompt='input path directory to load', default='./',
              help='the directory of path to load')
@click.option('--classifier_load_dir', prompt='input classifier directory to load', default='./',
              help='the directory of classifier to load')
@click.option('--misclassified_samples_dir', prompt='input the dir to save the misclassified samples',
              help='input the dir to save the misclassified samples')
@click.option('--acc_samples_dir', prompt='input the dir to save the well-classified samples',
              help='input the dir to save the well-classified samples')
@click.option('--data_dir', prompt='input data directory to exec',
              help='the directory of dataset to exec')
@click.option('--data_type', prompt='input data type [MNIST/FASHION_MNIST/CIFAR10]',
              help='the type of dataset [MNIST/FASHION_MNIST/CIFAR10]')
@click.option('--is_test_set', prompt='input whether to execute the test set [yes/no]', default='yes',
              help='input whether to execute the test set')

def start_detect(model_dir, path_load_dir, classifier_load_dir, misclassified_samples_dir, acc_samples_dir, data_dir,
                 data_type, is_test_set):
    # Load the model
    net = torch.load(model_dir, map_location=torch.device('cpu'))

    # Set to evaluation mode
    net.eval()

    # Generate feature list for each convolutional layer
    net.apply(generate_layer)

    # Register hook functions
    net.apply(register_hook)

    try:
        # Load the data
        loadData(data_dir, data_type, False)
    except BaseException as e:
        print("\033[0;31;40m", e, "\033[0m")
        exit(1)

    # Load extracted path information
    load_path(path_load_dir)

    # Load the classifier
    clf = load_classifier(classifier_load_dir)

    # Perform detection
    detect(net, clf, is_test_set, misclassified_samples_dir, acc_samples_dir)


@click.command(name='data-provider-verify')
@click.option('--model_dir', prompt='input model directory', help='the directory of MUT')
@click.option('--path_load_dir', prompt='input path directory to load', default='./',
              help='the directory of path to load')
@click.option('--classifier_load_dir', prompt='input classifier directory to load', default='./',
              help='the directory of classifier to load')
@click.option('--misclassified_samples_dir', prompt='input the dir to save the misclassified samples',
              help='input the dir to save the misclassified samples')
@click.option('--acc_samples_dir', prompt='input the dir to save the well-classified samples',
              help='input the dir to save the well-classified samples')
@click.option('--data_dir', prompt='input data directory to exec',
              help='the directory of dataset to exec')
@click.option('--data_type', prompt='input data type [MNIST/FASHION_MNIST/CIFAR10]',
              help='the type of dataset [MNIST/FASHION_MNIST/CIFAR10]')

def data_provider_verify(model_dir, path_load_dir, classifier_load_dir, misclassified_samples_dir, acc_samples_dir,
                         data_dir, data_type):
    # Load the model
    net = torch.load(model_dir, map_location=torch.device('cpu'))

    # Set to evaluation mode
    net.eval()

    # Generate feature list for each convolutional layer
    net.apply(generate_layer)

    # Register hook functions
    net.apply(register_hook)

    try:
        # Load data using a data provider
        loadDataProvider(data_dir, data_type)
    except BaseException as e:
        print("\033[0;31;40m", e, "\033[0m")
        exit(1)

    # Load extracted path information
    load_path(path_load_dir)

    # Load the classifier
    clf = load_classifier(classifier_load_dir)

    # Perform verification
    verify(net, clf, misclassified_samples_dir, acc_samples_dir)



def verify(net, clf, misclassified_samples_dir, acc_samples_dir):
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Load image and label
            image, label = data
            image, label = image.to(device), label.to(device)

            # Get predictions from the model
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)

            # Print the model's prediction
            print(f"{data_name_list[i]} Original model prediction: {classes[predicted]}", end="  ")

            # Simulate progress bar or animations using buffer
            for b_idx in range(len(buffer)):
                print(f"\b{buffer[b_idx]}", end="")
                time.sleep(0.2)

            print(end=" ")

            # Aggregate features for the last sample in each layer
            ll = []
            for layer_th in range(len(layers)):
                ll.append(layers[layer_th][-1])
            res = torch.stack(tuple(ll))

            # Transpose the result
            item = res.t()

            # Prepare the feature vector for classification
            to_pred_cg_list = [item.reshape(-1).tolist()]
            preds = clf[predicted].predict(to_pred_cg_list)

            # Classification result
            if preds == 0:
                print("PSRP Detection Result: \033[0;31;40m Misclassified \033[0m")
                torchvision.utils.save_image(
                    dataloader_copy[i][0],
                    f"{misclassified_samples_dir}/mis_{i}_{provider_classes[label]}_{classes[predicted]}.jpg"
                )
            else:
                print("PSRP Detection Result: \033[1;32m Correctly Classified \033[0m")
                torchvision.utils.save_image(
                    dataloader_copy[i][0],
                    f"{acc_samples_dir}/acc_{i}_{provider_classes[label]}_{classes[predicted]}.jpg"
                )

            # Clear intermediate layer data
            clear_layer()



def detect(net, clf, is_test_set, misclassified_samples_dir, acc_samples_dir):
    # Select the appropriate data loader based on the test flag
    loader = testloader if is_test_set == 'yes' else trainloader
    loader_copy = testloader_copy if is_test_set == 'yes' else trainloader_copy

    with torch.no_grad():  # Disable gradient calculations for inference
        for i, data in enumerate(loader):
            # Get image and label from the data
            image, label = data
            image, label = image.to(device), label.to(device)

            # Model prediction
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score

            print(f"Original model prediction: {classes[predicted]}", end="  ")

            # Simulate progress or animation
            for b_idx in range(len(buffer)):
                print(f"\b{buffer[b_idx]}", end="")
                time.sleep(0.2)

            print(end=" ")

            # Aggregate features for the last sample in each layer
            ll = []
            for layer_th in range(len(layers)):
                ll.append(layers[layer_th][-1])
            res = torch.stack(tuple(ll))

            # Transpose the result to prepare for classification
            item = res.t()

            # Reshape and convert the features to a list for the classifier
            to_pred_cg_list = [item.reshape(-1).tolist()]
            preds = clf[predicted].predict(to_pred_cg_list)  # Use the classifier to make a prediction

            # Process results
            if preds == 0:
                print("Detection Result: \033[0;31;40m Misclassified \033[0m")
                # Save the misclassified sample's image
                torchvision.utils.save_image(
                    loader_copy[i][0],
                    f"{misclassified_samples_dir}/mis_{i}_{classes[label]}_{classes[predicted]}.jpg"
                )
            else:
                print("Detection Result: \033[1;32m Correctly Classified \033[0m")
                # Save the correctly classified sample's image
                torchvision.utils.save_image(
                    loader_copy[i][0],
                    f"{acc_samples_dir}/acc_{i}_{classes[label]}_{classes[predicted]}.jpg"
                )

            # Clear layer activations or intermediate data
            clear_layer()



def execute(net, clf, is_test_set):
    loader = testloader if is_test_set == 'yes' else trainloader
    collect = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    total = 0
    with tqdm(total=len(loader), desc='Executing, please wait', leave=True, unit='B', unit_scale=True) as pbar:
        with torch.no_grad():
            for i, data in enumerate(loader):
                image, label = data
                image, label = image.to(device), label.to(device)
                outputs = net(image)
                # predicted
                _, predicted = torch.max(outputs.data, 1)

                ll = []
                for layer_th in range(len(layers)):
                    ll.append(layers[layer_th][-1])
                res = torch.stack(tuple(ll))

                item = res.t()

                to_pred_cg_list = [item.reshape(-1).tolist()]
                preds = clf[predicted].predict(to_pred_cg_list)
                test_oracle = 1 if predicted == label else 0

                collect += 1 if preds == test_oracle else 0

                if preds == 0:
                    if test_oracle == 0:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if test_oracle == 1:
                        TN += 1
                    else:
                        FN += 1

                clear_layer()
                total += 1
                pbar.update(1)
        pbar.close()

        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        FPR = FP / (TN + FP)
        print('----------metrics-----------')
        print()
        table = PrettyTable(['A\\P', '正', '负'])
        table.add_row(['正', TP, FN])
        table.add_row(['负', FP, TN])
        print(table)
        print()
        print('precision = %.4f, recall = %.4f, FPR = %.4f' % (prec, recall, FPR))


def clear_layer():
    for idx in range(len(layers)):
        layers[idx].clear()


def load_classifier(classifier_load_dir):
    clf = []
    file = os.listdir(classifier_load_dir)
    file_list = list(filter(lambda item: '_standardscaler_' in item, file))
    c1 = file_list[0]
    classifier_type = c1[:c1.find('_')]
    for k in range(10):
        try:
            classifier_file = classifier_load_dir + '/' + classifier_type + '_standardscaler_' + str(k) + '.joblib'
            clf.append(load(classifier_file))
        except FileNotFoundError as e:
            print("\033[0;31;40m", e, "\033[0m")
            exit(1)

    clf = tuple(clf)
    return clf


# -----------Training the Classifier--------------
def to_train(classifier_dir, classifier_type, all_features_to_train):
    # True labels of the samples to be trained
    all_orig_lblist = torch.tensor(global_label_list)
    # Predicted labels of the samples to be trained
    all_orig_pdlist = torch.tensor(global_pred_list)

    with tqdm(total=10, desc='Training the classifier, please wait', leave=True, unit='B', unit_scale=True) as pbar:
        for k in range(10):
            to_save_filename = classifier_dir + '/' + classifier_type.lower() + '_standardscaler_' + str(k) + '.joblib'

            to_train_cg = all_features_to_train[all_orig_pdlist == k]
            to_train_lb = all_orig_lblist[all_orig_pdlist == k]
            to_train_cg_list = []
            for item in to_train_cg:
                to_train_cg_list.append(item.reshape(-1).tolist())

            # Construct training labels
            labels = np.ones(to_train_lb.shape[0])
            labels[to_train_lb != k] = 0

            # Create a binary classification model
            clf = None
            if classifier_type == 'SVM':
                # Support Vector Machine
                clf = make_pipeline(StandardScaler(), svm.SVC())
            elif classifier_type == 'RF':
                # Random Forest
                clf = make_pipeline(StandardScaler(), RandomForestClassifier())
            elif classifier_type == 'LR':
                # Logistic Regression
                clf = make_pipeline(StandardScaler(), LogisticRegression())
            elif classifier_type == 'DT':
                # Decision Tree
                clf = make_pipeline(StandardScaler(), tree.DecisionTreeClassifier())
            elif classifier_type == 'AB':
                # AdaBoost Classifier
                clf = make_pipeline(StandardScaler(), AdaBoostClassifier())
            elif classifier_type == 'GNB':
                # Gaussian Naive Bayes
                clf = make_pipeline(StandardScaler(), GaussianNB())
            else:
                print("\033[0;31;40m", 'Unsupported classifier type', "\033[0m")
                exit(1)

            clf.fit(to_train_cg_list, labels)
            dump(clf, to_save_filename)
            pbar.update(1)


def save_path():
    idx = 1
    for perm in perms:
        filename = global_path_save_dir + '/path' + str(idx) + '.pkl'
        torch.save(perm, filename)
        idx += 1


def load_path(path_load_dir):
    file = os.listdir(path_load_dir)
    file_length = len(list(filter(lambda item: item[:4] == 'path', file)))
    global path_count_to_extract
    with tqdm(total=file_length, desc='Loading the path, please wait', leave=True, unit='B', unit_scale=True) as pbar:
        for idx in range(file_length):
            filename = path_load_dir + '/path' + str(idx + 1) + '.pkl'
            try:
                path_info = torch.load(filename, map_location=torch.device('cpu'))
                perms.append(path_info)
                path_count_to_extract = len(path_info)
            except FileNotFoundError as e:
                print("\033[0;31;40m", e, "\033[0m")
                exit(1)
            pbar.update(1)


def load_features(features_load_dir):
    file = os.listdir(features_load_dir)
    file_length = len(list(filter(lambda item: item[:13] == 'feature_layer', file)))
    global global_pred_list
    global global_label_list
    with tqdm(total=file_length, desc='Loading features, please wait', leave=True, unit='B', unit_scale=True) as pbar:
        for idx in range(file_length):
            filename = features_load_dir + '/feature_layer' + str(idx + 1) + '.pkl'
            try:
                feature_info = torch.load(filename, map_location=torch.device('cpu'))
                layers.append(feature_info)
            except FileNotFoundError as e:
                print("\033[0;31;40m", e, "\033[0m")
                exit(1)
            pbar.update(1)
    try:
        global_pred_list = torch.load(features_load_dir + '/pred_list.pkl', map_location=torch.device('cpu'))
        global_label_list = torch.load(features_load_dir + '/label_list.pkl', map_location=torch.device('cpu'))
    except FileNotFoundError as e:
        print("\033[0;31;40m", e, "\033[0m")
        exit(1)


def save_features(pred_list, label_list):
    idx = 1
    for feature in layers:
        filename = global_features_save_dir + '/feature_layer' + str(idx) + '.pkl'
        torch.save(feature, filename)
        idx += 1

    torch.save(pred_list, global_features_save_dir + '/pred_list.pkl')
    torch.save(label_list, global_features_save_dir + '/label_list.pkl')


def register_hook(m):
    global th
    name = m.__class__.__name__
    if name.find('Conv') != -1:
        m.register_forward_hook(retHook(th))
        th += 1


def generate_perms(m):
    name = m.__class__.__name__
    if name.find('Conv') != -1:
        min_to_extract = min(path_count_to_extract, m.out_channels)
        perms.append(torch.randperm(m.out_channels)[:min_to_extract])


def generate_layer(m):
    name = m.__class__.__name__
    if name.find('Conv') != -1:
        layers.append([])


def retHook(th):
    def hook(module, input, output):
        temp = torch.mean(torch.mean(output, dim=2), dim=2)
        layers[th].append(temp[0][perms[th]])

    return hook


cli.add_command(download)
cli.add_command(path_extract)
cli.add_command(features_extract)
cli.add_command(train_binary_classifier)
cli.add_command(execute_result)
cli.add_command(start_detect)
cli.add_command(data_provider_verify)
cli()
