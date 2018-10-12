 
import os
import argparse
 

from torchvision import transforms

from torch.utils.data import Dataset
from torchvision import datasets
import torch
from torch import optim
from  net import VGG_networks
import torchvision.models as models

from torch.autograd import Variable 


import torch.nn as nn 
import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


 
#import helper

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))




parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16', choices=model_names, help='choose model architecture: ' +
                        ' | '.join(model_names) + ' (default: vgg16)')


parser.add_argument('--in_file', type=str,
                    help='input text file')

parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')

parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')

parser.add_argument('--num_epochs', type=int, default=5,
                    help='number of epochs')

parser.add_argument('--print_every', type=int, default=20,
                    help='print frequency')

parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')

parser.add_argument('--dropout_prob', type=float, default=0.7,
                    help='probability of dropping weights')

parser.add_argument('--gpu', action='store_true', default=False,
                    help='run the network on the GPU')


 
parser.add_argument('--init_from', type=str, default=None,
                    help='initialize network from checkpoint')

parser.add_argument('--hidden_unit', type=float, default=3,
                    help='Number of hidden units')




args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    raise OSError(f'Directory {args.save_dir} does not exist. Please ensure that you have one')

 



 

 



#setting up variables for params tweaking 

cuda = args.gpu

data_dir=args.in_file
model =   args.arch
lr= args.learning_rate
epoch = args.num_epochs
hidden_features=args.hidden_unit

cuda = args.gpu
#cuda= not args.gpu
#cuda = str2bool(args.gpu)
 


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
'''
data_transforms.update({'test': transforms.Compose([   transforms.Resize(256), transforms.CenterCrop(224),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), transforms.ToTensor() ])})
         
'''    
data_transforms = {
    
    
    'train': transforms.Compose([
        transforms.RandomRotation(45),

        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        
        
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}

# TODO: Load the datasets with ImageFolder
dirs = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}



# TODO: Using the image datasets and the trainforms, define the dataloaders
 
dataloaders = {x_feat: torch.utils.data.DataLoader(image_datasets[x_feat], batch_size=32,
                                              shuffle=True)
               for x_feat in ['train', 'valid', 'test']}






    
 
#pretained moddel

model = models.__dict__[model](pretrained=True)

#model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Create your own classifier
net = VGG_networks(25088, 4096, len(cat_to_name))

# Put your classifier on the pretrained network
model.classifier = net

# Defining the loss and optimizing the classifier 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)




# Preparing a function to depict the validation results
def valid_show(model, val_data, criterion, cuda=False):
    val_start = time.time()
    running_val_loss = 0
    accuracy = 0
    for inputs, labels in val_data:
        inputs, labels = Variable(inputs), Variable(labels)
        
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        else:
                            # Use cpu
            inputs, labels = inputs.cpu(), labels.cpu()            

        outputs = model.forward(inputs)
        val_loss = criterion(outputs, labels)

        ps = torch.exp(outputs.data)
        
        _, predicted = ps.max(dim=1)
        
        equals = predicted == labels.data
        accuracy += torch.sum(equals)/len(equals)

        running_val_loss += val_loss.data[0]
    val_time = time.time() - val_start
    print("Valid loss: {:.3f}".format(running_val_loss/len(dataloaders['valid'])),
          "Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])),
          "Val time: {:.3f} s/batch".format(val_time/len(dataloaders['valid'])))

epochs = epoch
#cuda = True
print_every_n = 20





if cuda:
    model.cuda()
    print ("gpu activated")
else:
    model.cpu()

    

    

    
model.train()
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}")
    counter = 0
    running_loss = 0
    for inputs, labels in dataloaders['train']:
        counter += 1
        
        # Training pass
        inputs, labels = Variable(inputs), Variable(labels)
        
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        else:
            
                # Use cpu
            inputs, labels = inputs.cpu(), labels.cpu()
        
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        
        if counter % print_every_n == 0:
            print(f"Step: {counter}")
            print(f"Training loss {running_loss/counter:.3f}")
            model.eval()
            valid_show(model, dataloaders['valid'], criterion, cuda=cuda)
            model.train()
    else:
        # Validation pass
        train_end = time.time()
        model.eval()
        valid_show(model, dataloaders['valid'], criterion, cuda=cuda)



#Saving model

model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'arch': 'vgg16', 'hidden': 4096, 'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}, 'app-Algorithm.pt')



