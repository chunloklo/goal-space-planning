import ZEO
import os
import time
import ZODB
import ZEO
import BTrees
import transaction

ZEO_PORT = 8200

def start_zeo_server():
    db_folder = f'{os.getcwd()}/experiment_db/'

    if not os.path.exists(db_folder):
        time.sleep(2)
        try:
            os.makedirs(db_folder)
        except:
            pass

    address, stop = ZEO.server(db_folder + 'zeo_experiments.fs', port=ZEO_PORT)

    return address, stop

def open_zeo_client():
    client = ZEO.client(ZEO_PORT, cache_size = 0)
    db = ZODB.DB(client)
    connection = db.open()
    root = connection.root
    return root

def zodb_saver(obj, folder, run_hash):
    root = open_zeo_client()

    if not hasattr(root, 'experiments'):
        root.experiments = BTrees.OOBTree.BTree()

    if not hasattr(root.experiments, folder):
        root.experiments[folder] = BTrees.OOBTree.BTree()

    root.experiments[folder][run_hash] = obj
    transaction.commit()

def zodb_loader(folder, run_hash):
    root = open_zeo_client()
    return root.experiments[folder][run_hash]