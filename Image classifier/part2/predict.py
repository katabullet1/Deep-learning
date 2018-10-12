



import argparse
import torchvision.models as models
import numpy as np
from PIL import Image
import json
import torch
from  net import VGG_networks

from torch.autograd import Variable
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', type=str, default=None,
                    help='Path to input image')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Load checkpoint for prediction')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Run the network on a GPU')
 
parser.add_argument('--category_names', type=str, default=None,
                    help='Path to JSON file mapping categories to names')

parser.add_argument('--top_k', type=int, default=5,  help='show top probabilities')

args = parser.parse_args()
 
if not args.checkpoint:

    raise argparse.ArgumentTypeError('Load your saved classifier here')



cuda=args.gpu
topk=args.top_k


#preparing names for flowers 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Create the classifier
    net = VGG_networks(25088, checkpoint['hidden'], len(model.class_to_idx))

    # Put the classifier on the pretrained network
    model.classifier = net
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
model = load_checkpoint('app-Algorithm.pt')





 


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    
    aspect = image.size[0]/image.size[1]
    if aspect > 0:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    left_margin = (image.width-224)/2
    top_margin = (image.height-224)/2
    image = image.crop((left_margin, top_margin, left_margin+224, top_margin+224))
    
    # Now normalize...
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))
    
    return image
#location

 
image_path = 'flowers/test/2/image_05109.jpg'
image = Image.open(image_path)
image = process_image(image)
 


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=topk, cuda=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    inputs = Variable(image_tensor, requires_grad=False)

    if cuda:
        inputs = inputs.cuda()
 
    else:
    
        inputs=inputs.cpu()    

    inputs = inputs.unsqueeze(0)
    

    ps = torch.exp(model.forward(inputs))

    top_probs, top_labels = ps.topk(topk)
    top_probs, top_labels = top_probs.data.numpy().squeeze(), top_labels.data.numpy().squeeze()
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in top_labels]

    top_names=[cat_to_name[each] for each in top_classes]

    return top_probs, top_names




 






image_path = 'flowers/test/2/image_05109.jpg'

if cuda:
    model.eval()
    print("gpu active",predict(image_path, model))
else:
    model.eval()
    print(predict(image_path, model))







#top k

def predict(image_path, model, topk=5, cuda=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    inputs = Variable(image_tensor, requires_grad=False)

    if cuda:
        inputs = inputs.cuda()
 
    else:
    
        inputs=inputs.cpu()    

    inputs = inputs.unsqueeze(0)
    

    ps = torch.exp(model.forward(inputs))

    top_probs, top_labels = ps.topk(topk)
    top_probs, top_labels = top_probs.data.numpy().squeeze(), top_labels.data.numpy().squeeze()
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in top_labels]

 

    return top_probs, top_classes






image_path = 'flowers/test/2/image_05109.jpg'

model.eval()
top_probs, top_classes = predict(image_path, model)
image = Image.open(image_path)
image = np.array(image)

fig, (img_ax, p_ax) = plt.subplots(figsize=(4,7), nrows=2)
img_ax.imshow(image)   
img_ax.xaxis.set_visible(False)
img_ax.yaxis.set_visible(False)

p_ax.barh(np.arange(5, 0, -1), top_probs)
top_cat_names = [cat_to_name[each] for each in top_classes]
p_ax.set_yticks(range(1,6))
p_ax.set_yticklabels(reversed(top_cat_names));
fig.tight_layout(pad=0.1, h_pad=0)

plt.show()