from time import time

import torch
import pandas as pd
from pathlib import Path
import sklearn.preprocessing
import numpy as np
import onlinehd

def load():
    # fetches data
    data_dir = Path('ISOLET/data/')
    trn_file = data_dir / 'isolet1+2+3+4.data'
    tst_file = data_dir / 'isolet5.data'

    trn = pd.read_csv(trn_file, index_col=0)
    tst = pd.read_csv(tst_file, index_col=0)

    x = trn.values
    y = trn.iloc[:, -1].values

    x_test = tst.values
    y_test = tst.iloc[:, -1].values

    # normalize
    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    y = np.array(y)
    y_test = np.array(y_test)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long() - 1
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long() - 1
    return x, x_test, y, y_test

# simple OnlineHD training
def main(param):
    print("----------try: ", param.get('lr'), param.get('epoch'), param.get('dim'), param.get('bootstrap'))

    print('Loading...')
    x, x_test, y, y_test = load()
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features, dim=param.get('dim'))

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        model = model.to('cuda')
        print('Using GPU!')

    print('Training...')
    t = time()
    model = model.fit(x, y, bootstrap=param.get('bootstrap'), lr=param.get('lr'), epochs=param.get('epoch'))
    t = time() - t

    print('Validating...')
    yhat = model(x)
    yhat_test = model(x_test)
    acc = (y == yhat).float().mean()
    acc_test = (y_test == yhat_test).float().mean()
    print(f'{acc = :6f}')
    print(f'{acc_test = :6f}')
    print(f'{t = :6f}')

    return(format(acc, ".6f"), format(acc_test, ".6f"), format(t, ".6f"))

# trying kinds of hyperparameters and save them
result_file = open("ISOLET/result/hyperparameters.csv", "w")
result_file.write('lr, epoch, dim, bootstrap, acc, acc_test, t \n')

def hyperparameters():
    params = {'lrs': [0.2, 0.3, 0.4, 0.5], 'epochs': [20, 40, 60], 'dimensions': [5000, 7500, 10000],'bootstraps': [0.25, 0.5]}

    for lr in params.get('lrs'):
        for epoch in params.get('epochs'):
            for dim in params.get('dimensions'):
                for bootstrap in params.get('bootstraps'):
                    param = {'lr': lr, 'epoch':epoch, 'dim':dim, 'bootstrap': bootstrap}
                    acc, acc_test, t = main(param)

                    result_file.write(str(lr)+','+str(epoch)+','+str(dim)+','+str(bootstrap)+','+str(acc)+','+str(acc_test)+','+str(t)+'\n')

if __name__ == '__main__':
    hyperparameters()

result_file.close()