import os
import sys
import subprocess

sys.path.append(os.getcwd())
from src.utils import globals
from src.data_management import zeo_common

if __name__ == "__main__":
    address, stop = zeo_common.start_zeo_server()
    os.environ["USE_ZODB"] = "TRUE"
    subprocess.run(sys.argv[1:])
    stop()

