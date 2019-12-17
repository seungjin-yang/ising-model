import os
import glob
from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd


CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


def load_data(path):
    npz_file = np.load(path)
    susceptibility = float(npz_file['susceptibility'])

    basename = os.path.basename(path).rstrip('.npz')
    metadata = [each.split('-') for each in basename.split('_')]
    metadata = {key: value for key, value in metadata}
    T = float(metadata['ToverTc']) * CRITICAL_TEMPERATURE
    L = int(metadata['L'])
    return L, T, susceptibility


def parabola(x, a, b, c):
    return c * (x - a)**2 + b


def get_maximum(T, X):
    argmax = X.argmax()
    p0 = [T[argmax], X[argmax], -10]
    fit_range = slice(argmax - 1, argmax + 2)

    popt, _ = curve_fit(
        f=parabola,
        xdata=T[fit_range],
        ydata=X[fit_range],
        p0=p0)

    T_fit = np.linspace(T.min(), T.max(), 1000)
    X_fit = parabola(T_fit, *popt)

    argmax_fit = X_fit.argmax()
    return T_fit[argmax_fit], X_fit[argmax_fit]



def main():
    df = []
    for each in tqdm(glob.glob('./data/*.npz')):
        L, T, X = load_data(each)
        df.append([L, T, X])
    df = pd.DataFrame(df, columns=['L', 'T', 'X'])

    for each in df['L'].unique():
        df_with_same_L = df[df['L'] == each]
        T = df_with_same_L['T'].values
        X = df_with_same_L['X'].values

        sorting_indices = T.argsort()
        T = T[sorting_indices]
        X = X[sorting_indices]

        try:
            T_star, X_start = get_maximum(T, X)
        except:
            T_star = -1.0
            X_star = -1.0

        print(f'(L, T*, X*) = ({each}, {T_star:.4f}, {X_start:.4f})')

    print(f'T_c: {CRITICAL_TEMPERATURE:.4f}')





if __name__ == '__main__':
    main()
