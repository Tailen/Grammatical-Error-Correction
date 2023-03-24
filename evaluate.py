import numpy as np
import torch
import os.path
from datasets import load_dataset


'''
Output: (precision, recall, f1) on CoNLL14 test set using provided m2scorer
'''
def evalCoNLL14():
    pass

'''
Reference corrected sentences are hidden and model output needs to be submitted to
https://codalab.lisn.upsaclay.fr/competitions/4057#participate
'''
def evalBEA19():
    pass

'''
Output: GLEU score on JFLEG test set using
https://www.nltk.org/api/nltk.translate.gleu_score.html
Dataset collected from https://huggingface.co/datasets/jfleg
'''
def evalJFLEG():
    eval_dataset = load_dataset("jfleg", split='test[:]')
    pass
