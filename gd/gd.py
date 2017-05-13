import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def f(x):
    return (x - 2) ** 2 - 3

def df_dx(x):
    return 2 * (x - 2)

EPS = 1e-1
BASE_LR = 1e-1

def main():
    x_i = 0
    it = 0
    #delta = 2*EPS
    while abs(df_dx(x_i)) > EPS:
        
        print('x_i = {}'.format(x_i))
        k = df_dx(x_i)
        b = f(x_i) - k * x_i
        grad = BASE_LR * k
        x_i -= grad
        
        def line(x):
            return k * x + b

        plt.plot([-1, 3], [line(-1), line(3)])
        plt.savefig('fig_{}.png'.format(it))
        it += 1

    print('Root: {}'.format(x_i))

if __name__ == '__main__':
  main()
