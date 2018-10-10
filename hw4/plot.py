import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas


parser = argparse.ArgumentParser()
parser.add_argument('--exps',  nargs='+', type=str)
parser.add_argument('--save', type=str, default=None)
args = parser.parse_args()

f, ax = plt.subplots(1, 1)
for i, exp in enumerate(args.exps):
    log_fname = os.path.join('data', exp, 'log.csv')
    csv = pandas.read_csv(log_fname)

    color = cm.viridis(i / float(len(args.exps)))
    ax.plot(csv['Itr'], csv['ReturnAvg'], color=color, label=exp)
    ax.fill_between(csv['Itr'], csv['ReturnAvg'] - csv['ReturnStd'], csv['ReturnAvg'] + csv['ReturnStd'],
                    color=color, alpha=0.2)

ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Return')

if args.save:
    os.makedirs('plots', exist_ok=True)
    f.savefig(os.path.join('plots', args.save + '.jpg'))
else:
    plt.show()
