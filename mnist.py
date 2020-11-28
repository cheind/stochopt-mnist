from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dists
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

BATCH = 16
NSAMPLES = 1024
SIGMA = 0.2
BHALF = BATCH // 2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    ds_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    ds_test = datasets.MNIST('../data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=BATCH, shuffle=False, drop_last=True)

    return train_loader, test_loader

def train(net, opt, dl_train, dev):
    net.train()
    for batch_idx, (data, target) in enumerate(dl_train):
        data = data.to(dev)
        target = target.to(dev)
        zhat = target[:BHALF] + target[BHALF:]
        zhat = zhat.unsqueeze(0).repeat(NSAMPLES, 1).float()
        
        logits = net(data)
        d = dists.Categorical(logits=logits)
        s = d.sample(sample_shape=(NSAMPLES,)) # NxBATCH
        x,y = s[..., :BHALF], s[...,BHALF:]        
        z = x + y
        logp = d.log_prob(s)
        logpx, logpy = logp[..., :BHALF], logp[...,BHALF:]

        reward = torch.exp(-(z-zhat)**2/(2*SIGMA**2))
        loss = -(logpx + logpy)*reward.detach()
        loss = loss.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch_idx % 20 == 0:
            print(loss.item(), reward.mean().item())

    return loss

def test(net, dl_test, dev):
    net.eval()
    N = 0
    K = 0
    with torch.no_grad():
        for data, target in dl_test:
            data, target = data.to(dev), target.to(dev)

            zhat = target[:BHALF] + target[BHALF:]

            probs = net(data)
            pred = probs.argmax(dim=1)
            z = pred[:BHALF] + pred[BHALF:]
            N += BHALF
            K += (zhat == z).sum().item()

    return K/N

def pretrain_net(net, opt, dl_train, dev):
    net.train()
    for batch_idx, (data, target) in enumerate(dl_train):
        data = data.to(dev)
        target = target.to(dev)

        logits = net(data)
        loss = F.cross_entropy(logits, target)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch_idx > 10:
            break



def main():
    dl_train, dl_test = load_data()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net().to(dev)
    opt = optim.SGD(net.parameters(), lr=1e-3, momentum=0.95)
    pretrain_net(net, opt, dl_train, dev)
    print('testing',test(net, dl_test, dev))
    #scheduler = StepLR(opt, step_size=1, gamma=0.7)
    for i in range(10):
        loss = train(net, opt, dl_train, dev)
        print('testing',test(net, dl_test, dev))
        #scheduler.step()

if __name__ == '__main__':
    main()