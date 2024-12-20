import os
import sys
import pathlib
import logging

from decouple import config

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
ARTIFACTS_PATH = os.path.join(PATH, 'artifacts')

LOG_TESTING = config('LOG_TESTING', cast=bool, default=True)
LOG = logging.getLogger('Testing')
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.NullHandler())
if LOG_TESTING:
    LOG.addHandler(logging.StreamHandler(sys.stdout))