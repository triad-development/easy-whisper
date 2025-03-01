import logging

# configure base logger
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

# export all modules
from easy_whisper.data_loader import *
from easy_whisper.data_loader_hf import *
from easy_whisper.hf_login import *
from easy_whisper.trainer import *
