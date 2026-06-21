from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optimizers
from torch.nn import CrossEntropyLoss
from alexnet import AlexNet
import torch, os

DEVICE  = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
DOWNLOAD_LINK = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz" 
ALPHA = 1e-5

transform = v2.Compose(
    [
        v2.Resize((227,227)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ## these are the means and std of imagenet alexnet
    ]
)
## CUSTOM LINKS WITH BETTER SPEED THAN PYTORCH
if DOWNLOAD_LINK:
    datasets.CIFAR10.url = DOWNLOAD_LINK

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset , batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64 , shuffle=False)


model = AlexNet(alpha=ALPHA).to(DEVICE)
if os.path.exists("./model.pth"):
    model.load_state_dict(torch.load("model.pth"))


optimizer = optimizers.Adam(model.parameters(), lr=ALPHA)
loss_fx = CrossEntropyLoss()

def train(dataloader, model , loss_fx , optimizer, verbose=True):
    size = len(dataloader.dataset)
    print(size)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fx(pred , y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss , correct = 0 , 0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")