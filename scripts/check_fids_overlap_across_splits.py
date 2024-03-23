#!/usr/bin/env python3

"""
Prints the overlap file ID (row_ID, sent ID) overlap of train/dev/test across 5 splits
"""

import sys
from pylibs.misc.io import read_simple_flist

if len(sys.argv) != 2:
    print('- usage : python3', sys.argv[0], 'xxlang_set-name_split_1.ixs')
    print('- eg 1  : python3', sys.argv[0], 'MLDoc/random_ixs/split_1/chinese_train_1000.fids')
    print('- eg 2  : python3', sys.argv[0], 'indicnlp-news-articles/random_indices/split_1/bn_train_row.ixs')
    sys.exit()

# splits
for i in range(1, 6, 1):
    fids1 = read_simple_flist(sys.argv[1].replace("split_1", f"split_{i}"))
    for j in range(i+1, 6, 1):
        fids2 = read_simple_flist(sys.argv[1].replace("split_1", f"split_{j}"))
        print(i, 'n', j, ":", len(set(fids1) & set(fids2)), '/', len(fids1))
