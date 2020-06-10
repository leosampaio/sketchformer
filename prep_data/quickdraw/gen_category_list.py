#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_category_list.py
Created on Aug 16 2019 11:53
Generate a list of class names for quickdraw

@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pandas as pd

IN = '/scratch/mnt/manfred/scratch/Tu/db/quickdraw/ndjson_simplified'
OUT = 'list_quickdraw.txt'
if __name__ == '__main__':
    flist = os.listdir(IN)
    classes = []
    for f in flist:
        if os.path.isfile(os.path.join(IN, f)):
            classes.append(f.split('.')[0])
    classes.sort()
    lst = pd.DataFrame(data={'category': classes})
    lst.to_csv(OUT, index=None, header=None)
