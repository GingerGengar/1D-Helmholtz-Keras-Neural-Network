"""
    some utility routines
"""

from pathlib import Path
from shutil import copy
from datetime import datetime

def is_file(file_name):
    p = Path(file_name)
    return p.is_file()

def is_dir(file_name):
    p = Path(file_name)
    return p.is_dir()

def mkdir(file_name, parent_flag=False):
    p = Path(file_name)
    if not p.is_dir():
        p.mkdir(parents=parent_flag)

def backup_file(file_name):
    p = Path(file_name)
    if p.is_file():
        bak_fname = file_name + ".bak"
        p_bak = Path(bak_fname)
        p.rename(bak_fname)

def rename_file(file_name, new_filename):
    p = Path(file_name)
    if p.is_file():
        p.rename(new_filename)

def copy_file(src_file, dest_file):
    copy(src_file, dest_file)

def get_timestamp():
    """
    generate time stamp string for current data/time
    :return: string containing current time stamp
    """
    time_stamp = datetime.now().strftime("_%m-%d-%Y_%I-%M-%S_%p")
    return time_stamp

