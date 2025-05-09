from datetime import datetime
from torch import nn


def log(*args):
    print(f'[{datetime.now()}]', *args)
    log_message = f'[{datetime.now()}] ' + ' '.join(map(str, args)) + '\n'

    with open('./Log/Log.txt', 'a') as f:
        f.write(log_message)
