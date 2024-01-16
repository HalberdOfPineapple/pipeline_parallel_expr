import logging
import os
import numpy as np
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
GPT_LOG_DIR = os.path.join(BASE_DIR, 'gpt_logs')
SPEED_LOG_DIR = os.path.join(BASE_DIR, 'speed_logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figs')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(SPEED_LOG_DIR): os.makedirs(SPEED_LOG_DIR)
if not os.path.exists(GPT_LOG_DIR): os.makedirs(GPT_LOG_DIR)
if not os.path.exists(FIG_DIR): os.makedirs(FIG_DIR)

LOGGER: logging.Logger = None
EXPR_NAME: str = None

def init_logger(expr_name: str, speed_log: bool=False, gpt_log: bool=False):
    # Setup logging
    print(f"Initialize logger for experiment: {expr_name}")

    log_dir = SPEED_LOG_DIR if speed_log else LOG_DIR
    log_dir = GPT_LOG_DIR if gpt_log else log_dir
    log_filename = os.path.join(log_dir, f'{expr_name}.log')
    
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the logger
    global LOGGER
    LOGGER = logging.getLogger(expr_name)

def get_logger():
    global LOGGER
    assert LOGGER is not None, "[utils.py] the global ogger has not been initialized. Please call init_logger() first."
    return LOGGER

def print_log(msg: str):
    print(msg)
    global LOGGER
    if LOGGER is not None:
        LOGGER.info(msg)

def set_expr_name(expr_name: str):
    global EXPR_NAME
    EXPR_NAME = expr_name

def get_expr_name():
    global EXPR_NAME
    return EXPR_NAME