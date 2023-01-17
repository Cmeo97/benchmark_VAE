import torch
import json
from torch.nn import functional as F
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
class downstream_task_classification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(downstream_task_classification, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

ap = argparse.ArgumentParser()
ap.add_argument(
    "--dataset",
    help='encoder celeba',
    type=bool,
    default=False,
    choices=[True, False],
)

ap.add_argument(
    "--model_name",
    help='decoder celeba',
    type=bool,
    default=False,
    choices=[True, False],
)

args = ap.parse_args()


def main(args):

            
    if args.dataset == "celeba":
        if args.enc_celeba:
            from pythae.models.nn.benchmarks.celeba import Encoder_Conv_VAE_CELEBA as Encoder_VAE
        else:
            from pythae.models.nn.benchmarks.shapes import Encoder_Conv_VAE_3DSHAPES as Encoder_VAE
        if args.dec_celeba:
            from pythae.models.nn.benchmarks.celeba import Decoder_Conv_VAE_CELEBA as Decoder_VAE
        else:
            from pythae.models.nn.benchmarks.shapes import SBD_Conv_VAE_3DSHAPES as Decoder_VAE

        # Spatial size of training images, images are resized to this size.
        image_size = 64
        img_folder=args.data_path+'celeba/img_align_celeba'
        # Transformations to be applied to each individual image sample
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
        # Load the dataset from file and apply transformations
        data = CelebADataset(f'{img_folder}/img_align_celeba', transform)
        train_data = np.zeros((162770, 3, 64, 64), float)
        eval_data = np.zeros((182637 - 162770, 3, 64, 64), float)
        for i in range(162770):
            train_data[i] = data[i]
        for j in range(182637 - 162770):
            eval_data[j] = data[162770 + j]
        print('data loading done!')

    if args.dataset == "cifar10":

       
        from pythae.models.nn.benchmarks.cifar10 import Encoder_Conv_VAE_CIFAR10 as Encoder_VAE

        if args.dec_celeba:
            from pythae.models.nn.benchmarks.cifar10 import Decoder_Conv_VAE_CIFAR10 as Decoder_VAE
        else:
            from pythae.models.nn.benchmarks.cifar10 import SBD_Conv_VAE_CIFAR10 as Decoder_VAE

        image_size=64
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(image_size)])

        train_data = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform).data.transpose((0, 3, 1, 2))/255
    

        eval_data = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform).data.transpose((0, 3, 1, 2))/255
       

        #classes = ('plane', 'car', 'bird', 'cat',
        #           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        



# Load the pre-trained model
model_path = 'path/to/your/model.pth'
loaded_model = torch.load(model_path)
loaded_model.eval()

# Instantiate the downstream task classifier
classifier = downstream_task_classification(input_size=N, num_classes=M)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(classifier.parameters())
loss_fn = nn.CrossEntropyLoss()

# Prepare the data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        # Compute the input for the classifier
        input_latent = loaded_model(inputs)
        output = classifier(input_latent)
        loss = loss_fn(torch.sigmoid(output), labels)
        loss.backward()
        optimizer.step()

# Evaluation on validation dataset
classifier.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in val_dataloader:
        labels = (labels + 1) / 2
        inputs_latent = loaded_model(inputs)
        outputs = classifier(inputs_latent)
        predicted = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f'Validation Acc: {val_acc:.4f}')

# Save the final accuracy in a JSON file
results = {'loaded_model_name': model_path, 'final_accuracy': val_acc}
if os.path.exists("downstream_task_results.json"):
    with open("downstream_task_results.json", "r") as jsonFile:
        data = json.load(jsonFile)
    data.append(results)
    with open("downstream_task_results.json", "w") as jsonFile:
        json.dump(data, jsonFile)
else:
    with open("downstream_task_results.json", "w") as jsonFile:
        json.dump([results], jsonFile)
