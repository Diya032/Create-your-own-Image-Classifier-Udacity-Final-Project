import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import time

from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'mobilenet_v2'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='2')
    parser.add_argument('--gpu', action='store', default='cuda')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define data transforms for training, validation, and testing
    training_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testing_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets
    image_datasets = [datasets.ImageFolder(train_dir, transform=training_transforms),
                      datasets.ImageFolder(valid_dir, transform=validation_transforms),
                      datasets.ImageFolder(test_dir, transform=testing_transforms)]

    # Create dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]

    return dataloaders

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    print_every = 5
    device = torch.device("cuda" if gpu == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the GPU if available

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders[0]):  # 0 = train
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model
            optimizer.zero_grad()
            # ... rest of your training loop


            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in dataloaders[1]:
                        inputs, labels = inputs.to(gpu), labels.to(gpu)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(dataloaders[1]):.3f}.. "
                      f"Accuracy: {accuracy/len(dataloaders[1]):.3f}")
                running_loss = 0
                model.train()

   # end = time.time()
   # total_time = end - start
    #print("Model Trained in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))

def main():
    args = parse_args()
    
    data_dir = args.data_dir
    gpu = args.gpu
    dataloaders = load_data(data_dir)

    if args.arch == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        classifier = nn.Sequential(
            nn.Linear(1280, int(args.hidden_units)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(args.hidden_units), 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(
            nn.Linear(25088, int(args.hidden_units)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(args.hidden_units), 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    else:
        print("Invalid architecture. Supported architectures: 'mobilenet_v2' and 'vgg16'")
        return

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_to_idx = dataloaders[0].dataset.class_to_idx
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_to_idx
    path = args.save_dir
    save_checkpoint(path, model, optimizer, args, classifier)

if __name__ == "__main__":
    main()
