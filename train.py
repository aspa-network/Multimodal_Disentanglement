"""
Training script for DMD
"""
from disentangle import Disentanglement_NS

Disentanglement_NS(model_name='dns', dataset_name='mosi', model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train')
