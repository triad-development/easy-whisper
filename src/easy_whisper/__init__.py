import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s")
logger = logging.getLogger(__name__)
