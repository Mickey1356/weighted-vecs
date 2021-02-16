import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('out.csv')

    col_names = ['CV_SUB', 'CV_CUBIC', 'MLE', 'SRP']
    # col_names = ['CV_SUB', 'CV_CUBIC', 'SRP']
    # col_names = ['CV_CUBIC','SRP']

    m_data = data.groupby('K')[col_names].mean()

    m_data.plot()
    plt.show()