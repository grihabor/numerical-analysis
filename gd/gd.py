import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def f(x):
    return 2 * (x - 2) ** 2 - 3

def df_dx(x):
    return 4 * (x - 2)

EPS = 1e-2
BASE_LR = 1e-1

ROI_LEFT = -.5
ROI_RIGHT = 2.5
ROI_BOTTOM = -5 
ROI_TOP = 8

def draw_f():
    x = np.arange(ROI_LEFT, ROI_RIGHT + 0.1, 0.1)
    print(x)
    y = np.array(list(map(f, x)))
    print(y)
    plt.plot(x, y, 'b')

def draw_arrow(x, y, arrow_len):
    plt.plot(
        [x] + [x + arrow_len, x + 0.8 * arrow_len] * 2, 
        [y, y, y - 0.2 * arrow_len, y, y + 0.2 * arrow_len],
        'g'
    )
    plt.plot(
        [x, x], [y, ROI_BOTTOM], 'g--'
    )


def main():
    x_list = []
    x_i = 0
    it = 0
    
    def condition(x):
        return abs(df_dx(x)) > EPS

    while condition(x_i):
        x_list.append(x_i)
        print('x_i = {}'.format(x_i))
        k = df_dx(x_i)
        b = f(x_i) - k * x_i
        grad = BASE_LR * k
        
        def line(x):
            return k * x + b

        def save_figure(filename):
            plt.xlim([ROI_LEFT, ROI_RIGHT])
            plt.ylim([ROI_BOTTOM, ROI_TOP])
            draw_f()
            draw_arrow(x_i, f(x_i), -grad)
            plt.plot([ROI_LEFT, ROI_RIGHT], [line(ROI_LEFT), line(ROI_RIGHT)], 'r')
            plt.plot(x_list, list(map(f, x_list)), 'go')
            plt.savefig(filename, bbox_inches='tight', dpi=200)
            plt.clf()

        if it < 5 or not condition(x_i - grad):
            save_figure('fig_{}.png'.format(it))

        it += 1
        x_i -= grad

    print('Root: {}'.format(x_i))

if __name__ == '__main__':
  main()
