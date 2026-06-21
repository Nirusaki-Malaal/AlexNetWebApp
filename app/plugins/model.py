import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.optim as optimizers
from torch.nn import CrossEntropyLoss

DEVICE  = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
DOWNLOAD_LINK = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
CLASSES = ["Airplane" , "Automobile" , "Bird" , "Cat", "Deer" , "Dog" , "Frog", "Horse", "Ship" , "Truck"]
if DOWNLOAD_LINK:
    datasets.CIFAR10.url = DOWNLOAD_LINK


class Model(nn.Module):
    def __init__(self, num_classes=10, alpha=1e-4):

        ## ENABLING NN.MODULE TO REGISTER LAYERS
        super().__init__() 
        self.flatten = nn.Flatten()
        self.alpha = alpha
        
        # FEATURE EXTRACTOR
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # max pool 1
            nn.Conv2d(64,192, kernel_size=5, padding=2), # conv 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1), # conv 3
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256, kernel_size=3, padding=1), # conv 4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #conv 5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # max pooling

        )
        
        # FULLY CONNECTED LAYER
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        # TRANSFORM V2
        self.transform = transform = v2.Compose(
            [
                v2.Resize((227,227)),
                v2.ToImage(), ### converts torch vision image format ko convert karna to pytorch format
                v2.ToDtype(torch.float32, scale=True), ## converts the image back to tensor scaling it
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ## these are the means and std of image net
            ]
        )
        # TRAIN DATASET
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        
        # DATALOADER
        self.train_dataloader = DataLoader(train_dataset , batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=64 , shuffle=False)
        
        # GRADIENT DESCENTS AND LOSS F(X)
        self.optimizer = optimizers.Adam(self.parameters(), lr=self.alpha)
        self.loss_fx = CrossEntropyLoss()

    # FORWARD PROPAGATION
    def forward(self, x):
        x  = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    # TRAINING OF DATA
    def fit(self, verbose=True):
        size = len(self.train_dataloader.dataset)
        super().train()
        for batch, (X,y) in enumerate(self.train_dataloader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = self(X)
            loss = self.loss_fx(pred , y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                if verbose:
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self,verbose=True):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        super().eval()
        test_loss , correct = 0 , 0
        with torch.no_grad():
            for X,y in self.test_dataloader:
                X,y = X.to(DEVICE), y.to(DEVICE)
                pred = self(X)
                test_loss += self.loss_fx(pred,y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        if verbose:
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def fit_transform(self, epochs=10, verbose=True):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.fit(verbose)
            self.test(verbose)
        torch.save(self.state_dict(), "model.pth")
        if verbose:
            print("Saved PyTorch Model State to model.pth")
    

    def prediction(self, example):
        super().eval()
        with torch.no_grad(): ## disables the gradient calculation in pytorch
            example = example.to(DEVICE)
            logits = self(example) ## raw data
            print(logits)
            ## now apply softmax activation
            probablities = torch.nn.functional.softmax(logits, dim=1)
            max_class =  CLASSES[torch.argmax(probablities).item()] ## item() converts single element tensor to a scalar
        return max_class
    



