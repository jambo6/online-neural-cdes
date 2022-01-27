import time
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
import torchcde
import pandas as pd

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, width=128):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear0 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear1 = torch.nn.Linear(hidden_channels, width)
        self.linear2 = torch.nn.Linear(width, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear0(z)
        z = z.relu()

        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

    
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, 
                 interpolation="cubic", width=128, rectilinear=False):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels, width=width)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation
        self.rectilinear = rectilinear

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points)

        if self.rectilinear:
            pred_y = self.readout(z_T[:,0::2,:])
        else:
            pred_y = self.readout(z_T)
        return pred_y

    
def generate_bm(N=11, start=0, end=1, plot=False):
    '''
    Generate a standard Brownian path between start and end with N points.
    '''
    
    times = np.linspace(start, end, N)
    increment = (end-start)/(N-1)

    bm = np.zeros(N)
    for i in range(1,N):
        bm[i] = bm[i-1]+ np.random.normal(scale=np.sqrt(increment))

    if plot:
        plt.plot(times, bm)
        plt.title('BM')
        plt.show()

    return times, bm


def generate_labels(bm):
    '''
    Find whether the brownian path has is greater than 0 or not at terminal time

    '''
    return np.repeat(int(bm[-1]>0), len(bm))


def simulate_data(num_paths=100, N=50, start=0, end=1, plot=False):
    '''
    Generates Brownian paths
    '''
    
    X, Y = [], []
    for i in range(num_paths):
        t,x = generate_bm(N=N, start=start, end=end, plot=plot)
        y = generate_labels(x)

        if plot:
            plt.plot(t, x)

        X.append(np.array([t, x]).T)
        Y.append(y)

    return X, Y


def prepare_data_1D(X, Y):

    X2 = [torch.tensor(x) for x in X]
    Y2 = [torch.tensor(y) for y in Y]

    max_length = max([X2[i].size(0) for i in range(len(X2))])

    def fill_forward(x, max_length):
        return torch.cat([x, x[-1].unsqueeze(0).expand(max_length - x.size(0), x.size(1))])

    X3 = torch.stack([fill_forward(x, max_length=max_length) for x in X2])

    def fill_forward_labels(y, max_length):
        return torch.cat([y, y[-1].repeat(max_length - y.size(0))])

    Y3 = torch.stack([fill_forward_labels(y, max_length=max_length) for y in Y2])

    return X3.float(), Y3.float()


def get_data(num_paths=128):
    X, Y = simulate_data(num_paths=num_paths, N=3)
    X_tensor, Y_tensor = prepare_data_1D(X, Y)

    return X_tensor, Y_tensor


def main(num_paths=128, num_epochs=30, interp='cubic', hidden_channels=8, width=128,
         directory=None):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    train_X, train_y = get_data(num_paths=num_paths)
    train_X = train_X.to(device)
    train_y = train_y.to(device)

    if interp == 'rectilinear':
        model = NeuralCDE(input_channels=2, hidden_channels=hidden_channels, 
                          output_channels=1, interpolation='linear', width=width, 
                          rectilinear=True)
    elif interp == 'linear':
        model = NeuralCDE(input_channels=2, hidden_channels=hidden_channels, 
                          output_channels=1, interpolation='linear', width=width,)
    else:
        model = NeuralCDE(input_channels=2, hidden_channels=hidden_channels, 
                          output_channels=1, interpolation='cubic', width=width)       

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    if interp == 'cubic':
        train_coeffs = torchcde.natural_cubic_coeffs(train_X)
    elif interp == 'cubic_hermite':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)
    elif interp == 'rectilinear':
        train_coeffs = torchcde.linear_interpolation_coeffs(train_X, rectilinear=0)
    else:
        train_coeffs = torchcde.linear_interpolation_coeffs(train_X)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    test_X, test_y = get_data(num_paths=2**10)
    test_X = test_X.to(device)
    test_y = test_y.to(device)

    if interp == 'cubic':
        test_coeffs = torchcde.natural_cubic_coeffs(test_X)
    elif interp == 'cubic_hermite':
        test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    elif interp == 'rectilinear':
        test_coeffs = torchcde.linear_interpolation_coeffs(test_X, rectilinear=0)
    else:
        test_coeffs = torchcde.linear_interpolation_coeffs(test_X)
    pred_y = model(test_coeffs).squeeze(-1)

    test_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, test_y)
    print(f'Test MSE: {test_loss}')

    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.numel()
    print('Test Accuracy: {}'.format(proportion_correct))

    return train_X, train_y, test_X, test_y, pred_y, model

def find_prop(pred, labels, sigmoid=True):
    if sigmoid:
        binary_prediction = (torch.sigmoid(pred) > 0.5).to(labels.dtype)
    else:
        binary_prediction = (pred > 0.5).to(labels.dtype)

    prediction_matches = (binary_prediction == labels).to(labels.dtype)
    proportion_correct = prediction_matches.sum() / labels.numel()
    
    return proportion_correct

def find_acc(directories, sigmoid=True):
    device = torch.device("cpu")

    train_accu, test_accu = [], []
    for directory in directories:
        interp = directory.split('/')[-3]
        params = directory.split('/')[-2].split('_')
        params = [int(t) for t in params]
        print(params)
        num_epochs = params[0]
        num_paths = params[1]
        hidden_channels = params[2]
        width = params[3]

        if interp == 'rectilinear':
            model = NeuralCDE(input_channels=2, hidden_channels=hidden_channels, 
                              output_channels=1, interpolation='linear', width=width,
                              rectilinear=True)
        elif interp == 'linear':
            model = NeuralCDE(input_channels=2, hidden_channels=hidden_channels,
                              output_channels=1, interpolation='linear', width=width)
        else:
            model = NeuralCDE(input_channels=2, hidden_channels=hidden_channels, 
                              output_channels=1, interpolation='cubic', width=width)
            
        model.load_state_dict(torch.load(f'{directory}/model.pt'))
        model.to(device)

        train_x = torch.load(f'{directory}/train_x.pt')
        train_x = train_x.to(device)
        train_y = torch.load(f'{directory}/train_y.pt')
        train_y = train_y.to(device)
        test_x = torch.load(f'{directory}/test_x.pt')
        test_x = test_x.to(device)
        test_y = torch.load(f'{directory}/test_y.pt')
        test_y = test_y.to(device)
        pred_y = torch.load(f'{directory}/pred_y.pt')    
        pred_y = pred_y.to(device)

        if interp == 'cubic':
            train_coeffs = torchcde.natural_cubic_coeffs(train_x)
        elif interp == 'cubic_hermite':
            train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_x)
        elif interp == 'linear':
            train_coeffs = torchcde.linear_interpolation_coeffs(train_x)
        else:
            train_coeffs = torchcde.linear_interpolation_coeffs(train_x, rectilinear=0)

        pred = model(train_coeffs).squeeze(-1)

        train_prop = find_prop(pred, train_y)
        print('Train Accuracy: {}'.format(train_prop))
        
        test_prop = find_prop(pred_y, test_y)
        print('Test Accuracy: {}'.format(test_prop))
        
        train_accu.append(train_prop)
        test_accu.append(test_prop)
                
    return train_accu, test_accu
    

if __name__ == '__main__':

    config={
            'interp_methods': ['cubic', 'cubic_hermite', 'rectilinear', 'linear'],
            'num_epoch_vals': 100,
            'num_paths_vals': 2**12, 
            'hidden_channels_vals': 10,
            'widths': 256,
            }

    for interp in config['interp_methods']:
        for k in range(1,6):
            num_epochs = config['num_epoch_vals']
            num_paths = config['num_paths_vals']
            hidden_channels = config['hidden_channels_vals']
            width = config['widths']
            
            print(f'Interpolation: {interp}, num_epochs: {num_epochs}, num_paths: {num_paths}, hidden: {hidden_channels}, width: {width}')

            directory = f'../results/sim_bm/{interp}/{str(num_epochs)}_{str(num_paths)}_{str(hidden_channels)}_{str(width)}/{k}/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            start = time.time()
            train_x, train_y, test_x, test_y, pred_y, model = main(num_paths=num_paths, num_epochs=num_epochs, interp=interp, 
               hidden_channels=hidden_channels, width=width, directory=directory)
            end = time.time()
            print(f'Time elapsed is {end-start}')

            torch.save(train_x, f'{directory}train_x.pt')
            torch.save(train_y, f'{directory}train_y.pt')
            torch.save(test_x, f'{directory}test_x.pt')
            torch.save(test_y, f'{directory}test_y.pt')
            torch.save(pred_y, f'{directory}pred_y.pt')
            torch.save(model.state_dict(), f'{directory}model.pt')    
            
    # Get results
    
    directory1 = '../results/sim_bm/cubic'
    directory2 = '../results/sim_bm/cubic_hermite'
    directory3 = '../results/sim_bm/rectilinear'
    directory4 = '../results/sim_bm/linear'
    
    directories1 = [x[0] for x in os.walk(directory1)][2:]
    directories2 = [x[0] for x in os.walk(directory2)][2:]
    directories3 = [x[0] for x in os.walk(directory3)][2:]
    directories4 = [x[0] for x in os.walk(directory4)][2:]
    
    results = []
    for directories in [directories1, directories2, directories3, directories4]:
        interp = directories[0].split('/')[-3]
        train_accu, test_accu = find_acc(directories1)
        train_mean = np.mean(train_accu)
        train_sd = np.std(train_accu)
        test_mean = np.mean(test_accu)
        test_sd = np.std(test_accu)
        print(f'Mean train: {train_mean}, Mean test: {test_mean}')
        print(f's.d. train: {train_sd}, s.d. test: {test_sd}')
        results.append([interp, train_mean, train_sd, test_mean, test_sd])

    df = pd.DataFrame(results, columns=['interpolation', 'train_mean', 'train_sd', 
                                        'test_mean', 'test_sd'])
    df.to_csv('../results/sim_bm/results_table.csv', index=False)
            
