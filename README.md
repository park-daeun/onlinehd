# onlinehd
> UCI(University of California, Irvine) GREAT Program Project about Hyperdymensional Computing

<br>

## Setting

<details>
<summary><b>Read More</b></summary>
<div markdown="1">
<br>
   
**Authors**: Alejandro Hernández Cano, Mohsen Imani.

### Installation

In order to install the package, simply run the following:

```
pip install onlinehd
```

Visit the PyPI [project page](https://pypi.org/project/onlinehd/) for
more information about releases.
   
### Documentation

Read the [documentation](https://onlinehd.readthedocs.io/en/latest/)
of this project. 

### Quick start

The following code generates dummy data and trains a OnlnineHD classification
model with it.

```python
>>> import onlinehd
>>> dim = 10000
>>> n_samples = 1000
>>> features = 100
>>> classes = 5
>>> x = torch.randn(n_samples, features) # dummy data
>>> y = torch.randint(0, classes, [n_samples]) # dummy data
>>> model = onlinehd.OnlineHD(classes, features, dim=dim)
>>> if torch.cuda.is_available():
...     print('Training on GPU!')
...     model = model.to('cuda')
...     x = x.to('cuda')
...     y = y.to('cuda')
...
Training on GPU!
>>> model.fit(x, y, epochs=10)
>>> ypred = model(x)
>>> ypred.size()
torch.Size([1000])
```

For more examples, see the `example.py` script. Be aware that this script needs
`pytorch`, `sklearn` and `numpy` to run.

### Citation Request

If you use onlinehd code, please cite the following paper:

1. Alejandro Hernández-Cano, Namiko Matsumoto, Eric Ping, Mohsen Imani
   "OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using
   Hyperdimensional System", IEEE/ACM Design Automation and Test in Europe
   Conference (DATE), 2021.

</div>
</details>
<br>
<br>

## example.py
It learn the **MNIST dataset** which is pulled from scikit-learn
<br>
#### result
![image](https://user-images.githubusercontent.com/70877497/126602731-cb11bc70-105e-4996-a99b-88d68bdb26d6.png)
: testing accuracy and learning time (CPU)

<br>
<br>

## ISOLET.py
dataset: ISOLET https://archive.ics.uci.edu/ml/datasets/isolet
<br>(This data set was generated as follows. 150 subjects spoke the name of each letter of the alphabet twice. Hence, we have 52 training examples from each speaker. The speakers are grouped into sets of 30 speakers each, and are referred to as isolet1, isolet2, isolet3, isolet4, and isolet5. The data appears in isolet1+2+3+4.data in sequential order, first the speakers from isolet1, then isolet2, and so on. The test set, isolet5, is a separate file.)
<br>
#### result : [click](https://github.com/park-daeun/onlinehd/blob/main/ISOLET/result/hyperparameters.csv)

<br>
<br>
