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
    y = np.array(list(map(f, x)))
    plt.plot(x, y, 'b')

def draw_arrow(x, y, arrow_len=1, only_dot=False):
    if not only_dot:
        arrow = plt.plot(
            [x] + [x + arrow_len, x + 0.8 * arrow_len] * 2 + [x + arrow_len, x], 
            [y, y, y - 0.2 * (arrow_len + 2) / 4, y, y + 0.2 * (arrow_len + 2) / 4, y, y],
            'g'
        )
        plt.setp(arrow, linewidth=2)
    plt.plot(
        [x, x], [y, ROI_BOTTOM], 'g--'
    )


def main():
    x_list = []
    x_i = 0
    it = 0

    def special_cases():
        plt.xlim([ROI_LEFT, ROI_RIGHT])
        plt.ylim([ROI_BOTTOM, ROI_TOP])
        draw_f()
        plt.savefig('fig_func.png', bbox_inches='tight', dpi=200)
        plt.clf()

        plt.xlim([ROI_LEFT, ROI_RIGHT])
        plt.ylim([ROI_BOTTOM, ROI_TOP])
        draw_f()
        draw_arrow(x_i, f(x_i), only_dot=True)
        plt.plot([x_i], [f(x_i)], 'go')
        plt.savefig('fig_dot.png', bbox_inches='tight', dpi=200)
        plt.clf()

        
        plt.xlim([ROI_LEFT, ROI_RIGHT])
        plt.ylim([ROI_BOTTOM, ROI_TOP])
        draw_f()
        draw_arrow(x_i, f(x_i), only_dot=True)

        k = df_dx(x_i)
        b = f(x_i) - k * x_i

        def line(x):
            return k * x + b

        plt.plot([ROI_LEFT, ROI_RIGHT], [line(ROI_LEFT), line(ROI_RIGHT)], 'r')
        plt.plot([x_i], [f(x_i)], 'go')
        plt.savefig('fig_tangent.png', bbox_inches='tight', dpi=200)
        plt.clf()

    def condition(x):
        return abs(df_dx(x)) > EPS

    special_cases()

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
            print('Save figure: {}'.format(filename))
            plt.clf()

        if it < 5 or not condition(x_i - grad):
            filename = 'fig_{}.png'.format(it)
            save_figure(filename)


        it += 1
        x_i -= grad

    print('Minimum: {}'.format(x_i))

if __name__ == '__main__':
    main()
