
import logging
logger = logging.getLogger("nircls")
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
def log_versions():
    try:
        import numpy, pandas, sklearn
        logger.info(f"Versions | numpy {numpy.__version__} | pandas {pandas.__version__} | sklearn {sklearn.__version__}")
    except Exception as e:
        logger.warning(f"Version log failed: {e}")
