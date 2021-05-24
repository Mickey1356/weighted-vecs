import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('final/results/data/out_mnist.csv')
    # data = pd.read_csv('final/results/data/out_gisette.csv')

    # data = pd.read_csv('final/results/data/out_mnist_L.csv')
    # data = pd.read_csv('final/results/data/out_gisette_L.csv')


    col_names = ['CV_SUB', 'CV_CUBIC', 'MLE', 'SRP']
    # col_names = ['CV_CUBIC', 'MLE', 'SRP']
    # col_names = ['CV_SUB', 'CV_CUBIC', 'SRP']
    # col_names = ['CV_CUBIC','SRP']

    # m_data = data.groupby('K')[col_names].mean()
    m_data = data.groupby('K')[col_names].std(ddof=0)
    # print(m_data)

    # min_data = m_data.idxmin(axis=1)
    # print(min_data)

    m_data.plot()
    plt.title('Plot of std.dev against K (MNIST, small K)')
    plt.xlabel('Number of hashes K')
    plt.ylabel('Std. Dev.')
    plt.legend(fontsize=12)
    plt.show()