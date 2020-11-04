import pickle as pk
import numpy as np

for layer in range(0, 13):

    loaded = pk.load(open('predictions/_accs.pkl'.format(), 'rb'))
    mean_subj_acc_across_folds = loaded.mean(0)
