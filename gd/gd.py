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
LEFT = -1
RIGHT = 3

def draw_f():
    x = np.arange(-1, 3, 0.1)
    print(x)
    y = np.array(list(map(f, x)))
    print(y)
    plt.plot(x, y)

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

        plt.xlim([LEFT, RIGHT])
        plt.ylim([-6, 7])
        draw_f()
        plt.plot([LEFT, RIGHT], [line(LEFT), line(RIGHT)])
        plt.savefig('fig_{}.png'.format(it))
        plt.clf()

        it += 1

    print('Root: {}'.format(x_i))

if __name__ == '__main__':
  main()
