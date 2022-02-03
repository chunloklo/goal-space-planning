from ast import Index
from ctypes import sizeof
from distutils.command.config import config
from re import L
from sqlite3 import connect
import ZODB, ZODB.FileStorage
import os
from filelock import FileLock 
from BTrees import OOBTree
import transaction
from ZODB.blob import Blob, is_blob_record

zodb_connection = None

DB_DATA_KEY = 'data'
DB_CONFIGS_KEY = 'configs'
DB_FOLDER = 'results_dbs'

def open_db(db_folder, create_if_not_exist=True):
    if not os.path.isfile(f'{db_folder}/data.fs') and not create_if_not_exist:
        raise ValueError(f'Database at {db_folder} does not yet exist and open_db has been told not to create a database if it does not exit yet. Erroring out')

    os.makedirs(db_folder, exist_ok=True)
    lock_name = f'{db_folder}/access.lock'
    lock = FileLock(lock_name)
    lock.acquire()

    blob_dir = f'{db_folder}/blobs'

    storage = ZODB.FileStorage.FileStorage(f'{db_folder}/data.fs', blob_dir=blob_dir)
    db = ZODB.DB(storage)
    connection = db.open()
    root = connection.root()

    return db, connection, root, lock

def close_db(db, connection, lock):
    connection.close()
    db.close()
    lock.release()

def zodb_op_save(root, config_id, data, key):
    if key not in root:
        root[key] = OOBTree.OOBTree()

    root[key][config_id] = data

    transaction.commit()

def zodb_op_load(root, config_id, key):
    if key not in root:
        raise IndexError(f'{key} object is not in the database yet. Cannot retrieve any {key} for {config_id}')
        
    if config_id not in root[key]:
        raise IndexError(f'Cannot find config id {config_id} in {key} object')

    return root[key][config_id]

def save_config_and_data_zodb(config_id, config, data, db_folder):
    db, connection, root, lock = open_db(db_folder)
    zodb_op_save(root, config_id, config, DB_CONFIGS_KEY)
    zodb_op_save(root, config_id, data, DB_DATA_KEY)
    transaction.commit()
    close_db(db,  connection, lock)

def load_data_config_from_id(config_id, db_folder):
    global zodb_connection
    if zodb_connection is None:
        db, connection, root, lock = open_db(db_folder, create_if_not_exist=False)
    else:
        db, connection, root, lock = zodb_connection

    config = zodb_op_load(root, config_id, DB_CONFIGS_KEY)
    data = zodb_op_load(root, config_id, DB_DATA_KEY)

    if zodb_connection is None:
        close_db(db,  connection, lock)
    
    return config, data
    
class BatchDBAccess():
    """Used to open DB for batch retrieval access.
    Example: with BatchDBAccess(): # Retrieve data
    """
    def __init__(self, db_folder):
        self.db_folder = db_folder
    def __enter__(self):
        global zodb_connection
        zodb_connection = open_db(self.db_folder, create_if_not_exist=False)
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global zodb_connection
        # db, connection, and lock
        close_db(zodb_connection[0], zodb_connection[1], zodb_connection[3])
        zodb_connection = None