#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet tools

@author: Jarvis ZHANG
@date: 2017/8/15
@framework: Tensorflow
@editor: VS Code
"""

import os
import logging

def create_logger(log_file):
    """ Create a logger to write info to console and file.
    ### Params:
        * log_file - string: path to the logging file
    ### Return:
        A logger object of class getLogger()
    """
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    filehandler = logging.FileHandler(log_file)
    filehandler.setLevel(logging.INFO)
    
    # Create console handler
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s")
    filehandler.setFormatter(formatter)
    consolehandler.setFormatter(formatter)
    
    # Register both handlers
    logger.addHandler(filehandler)
    logger.addHandler(consolehandler)

    return logger