import os
import sys
import subprocess
import code

sys.path.append(os.getcwd())
from src.utils import globals
from src.data_management import zeo_common

def keys(btree):
    return list(btree.keys())

if __name__ == "__main__":
    globals.use_zodb = True
    address, stop = zeo_common.start_zeo_server()
    root = zeo_common.open_zeo_client()
    print('root should be your place!')
    code.interact(local=locals())


