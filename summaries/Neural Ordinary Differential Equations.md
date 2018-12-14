##### [18-12-12] [paper26]
- Neural Ordinary Differential Equations [[pdf]](https://arxiv.org/abs/1806.07366) [[code]](https://github.com/rtqichen/torchdiffeq) [[slides]](https://www.cs.toronto.edu/~duvenaud/talks/ode-talk-google.pdf) [[pdf with comments]](https://github.com/fregu856/papers/blob/master/commented_pdfs/Neural%20Ordinary%20Differential%20Equations.pdf)
- *Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud*
- `2018-10-22, NeurIPS2018`

****

### General comments on paper quality:
- Reasonably well-written but very interesting paper. The examples could have been more thoroughly explained, it feels like the authors probably struggled to stay within the page limit.

### Paper overview:
- The authors introduce a new family of deep neural networks by using black-box ODE solvers as a model component. 

- Instead of specifying a discrete sequence of hidden layers by: h_{t+1} = h_{t} + f(h_t, theta_t), where f is some neural network architecture, they interpret these iterative updates as an Euler discretization/approximation of the corresponding continuous dynamics and directly specify this ODE: dh(t)/dt = f(h(t), theta). To compute gradients, they use the adjoint method, which essentially entails solving a second, augmented ODE backwards in time.

- For example, if you remove the final ReLU layer, do not perform any down-sampling and have the same number of input- and output channels, a residual block in ResNet specifies a transformation precisely of the kind h_{t+1} = h_{t} + f(h_t, theta_t). Instead of stacking a number of these residual blocks, one could thus directly parameterize the corresponding ODE, dh(t)/dt = f(h(t), theta), and use an ODE solver to obtain h(T) as your output. The authors provide a [code example](https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py) where they replace 6 such residual blocks with this parameterized ODE in a larger classification model:
```
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + shortcut

class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.norm2 = norm(dim)
        self.conv2 = conv3x3(dim, dim)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
.
.
.
feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
.
.
.
model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
```

- When empirically evaluating this approach on MNIST, they find that the model using one ODEBlock instead of 6 ResBlocks achieves almost identical test accuracy, while using fewer parameters (parameters for just one block instead of six).

- The authors also apply their approach to density estimation and time-series modeling, but I chose to focus mainly on the ResNet example.
 
### Comments:
- Very interesting idea. Would be interesting to attempt to implement e.g. a ResNet101 using this approach. I guess one could try and keep the down-sampling layers, but otherwise replace layer1 - layer4 with one ODEBlock each? 

- It is however not at all clear to me how well this approach would scale to large-scale problems. Would it become too slow? Or would you lose accuracy/performance? Or would the training perhaps even become unstable? Definitely an interesting idea, but much more empirical evaluation is needed.
