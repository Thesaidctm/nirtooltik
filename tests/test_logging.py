import logging
from pathlib import Path

from nircls import cli
from nircls.utils.logging import logger


def test_log_file_created(tmp_path):
    cli.RESULTS_DIR = tmp_path / "results"
    log_file = tmp_path / "nircls.log"
    ret = cli.main(["--log-file", str(log_file), "demo"])
    assert ret == 0
    test_message = "injected error message"
    logger.error(test_message)
    for h in logging.getLogger().handlers:
        h.flush()
    assert log_file.exists()
    assert test_message in log_file.read_text()
