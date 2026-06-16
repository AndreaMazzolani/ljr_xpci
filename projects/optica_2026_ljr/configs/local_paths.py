"""Local machine paths for private data. This file is outside src because it is machine-specific configuration, not reusable package code."""

from pathlib import Path
import socket


DEFAULT_DATA_ROOTS = {
    ("clio",): Path("/home/amazzolani/Matlab_drive_scripts/remotedata_Andrea_clio"),
    ("zeus",): Path("/home/amazzolani/Matlab_drive_scripts/remotedata_Andrea_zeus"),
    ("mnemosyne", "mnem"): Path("/home/amazzolani/Matlab_drive_scripts/remotedata_Andrea_mnem"),
    ("dylan", "precision-5820", "work", "desk"): Path("/home/andrea/MATLAB_26_04_2022/Matlab_drive_scripts/remotedata_Andrea_desk"),
}


def get_default_data_root():
    hostname = socket.gethostname().lower()
    for machine_names, data_root in DEFAULT_DATA_ROOTS.items():
        if any(machine_name in hostname for machine_name in machine_names): return data_root
    return None



