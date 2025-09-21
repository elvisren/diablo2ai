import logging, os
def setup_logging(default_level: str = "INFO"):
    level = os.environ.get("APP_LOG_LEVEL", default_level).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
