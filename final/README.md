# Control Variates with 1 Extra Vector

## Abstract
We present an alternative technique for similarity estimation under locality sensitive hashing schemes. By utilising control variates, we are able to achieve better theoretical variance reductions compared to methods that rely on maximum likelihood estimation. We show that our method obtains equivalent results, but slight modifications can provide better empirical results at lower dimensions. Finally, we compare the various methods' performances on the MNIST and Gisette dataset, and show that our model achieves better accuracy and stability.

*Paper to be added at a later date.*

## Requirements
The code is written in Python3.

To install the required libraries, run `pip install -r requirements.txt`.

The main libraries used are `numpy` and `pytorch`. Other libraries have been imported for ease of use.

## Basic Instructions
1) First download Gisette data from https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/.

2) Run `mnist.py` to download MNIST dataset.

3) Run `gisette.py` to combine Gisette dataset

4) Create a folder called `data/` and place the generated files `mnist.pkl` and `gisette.npy` within

5) Run `compute_est_gpu.py`. In the `main` function, there are the following lines (lines: 288-292):
    ```python
    # mnist dataset
    data, (N, _) = load_mnist()
        
    # gisette dataset
    data, (N, _) = load_gisette()
    ```
    Comment out the respective line to either use the MNIST dataset or the Gisette dataset

    Other important lines are the following (also in `main`, lines 282-286):
    ```python
    # batch size depends on gpu memory
    # higher is better, to avoid the overhead from transferring from cpu to gpu as much as possible
    # for a 6gb gpu, mnist: 120000, gisette: 32000
    batch_size = 32000

    # set the size of dimensions to check
    Ks = range(10, 101, 10)
    # Ks = range(800, 1001, 50)
    ```

    Constants (which are in `ALL_CAPS`) can also be modified as needed.