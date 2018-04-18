# -*- coding: utf-8 -*-
"""Tensorboard interface for python with pycrayon

https://medium.com/@vishnups/pycrayon-tensorboard-741fbe05b348

Set up tensorboard
------------------

.. code-block:: bash

    pip install pycrayon
    docker pull alband/crayon
    # Tensorboard will be served on localhost:9118
    docker run -p 9118:8888 -p 9119:8889 --name crayon alband/crayon
"""
import collections
import datetime
import socket

import pycrayon


CrayonSettings = collections.namedtuple('CrayonSettings', ['host', 'port'])
CRAYON_SETTINGS = CrayonSettings(host='localhost', port='9119')


def get_experiments(names, settings=CRAYON_SETTINGS):
    """Creates pycrayon experiments object to log data to.
    """
    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
    client = get_crayon_client(settings=settings)
    return {
        each_name: client.create_experiment(
            '{name};{dt}_{host}'.format(
                dt=experiment_date,
                host=socket.gethostname(),
                name=each_name,
            )
        )
        for each_name in names
    }


def get_experiment(name, settings=CRAYON_SETTINGS):
    """Creates a pycrayon experiment object to log data to.
    """
    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
    experiment_name = '{name};{dt}_{host}'.format(
        dt=experiment_date,
        host=socket.gethostname(),
        name=name,
    )
    return get_crayon_client(settings=settings).create_experiment(experiment_name)


def get_crayon_client(settings=CRAYON_SETTINGS):
    return pycrayon.CrayonClient(hostname=settings.host, port=settings.port)


def clear_expts(settings=CRAYON_SETTINGS):
    get_crayon_client(settings=settings).remove_all_experiments()
