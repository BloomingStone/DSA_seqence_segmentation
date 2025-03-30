from typing import Optional, Callable
import logging
import time
from datetime import datetime
from pathlib import Path
from functools import wraps

# todo logger 本身就已经用单例实现了，改用名称或者__module__调用
_logger: Optional[logging.Logger] = None

def valid_init(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _logger
        if _logger is None:
            raise RuntimeError("Logger not initialized. Call init_logger first.")
        return func(*args, **kwargs)
    return wrapper

def init_logger(root_dir: Path) -> None:
    global _logger
    if _logger is not None:
        return

    _logger = logging.getLogger("log")
    _logger.setLevel(logging.DEBUG)

    # Handler of File
    assert root_dir.exists() and root_dir.is_dir()
    log_dir = root_dir / "log"
    log_dir.mkdir(exist_ok=True)
    handler_file = logging.FileHandler(
        filename=log_dir / f"train_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    )
    handler_file.setLevel(logging.INFO)

    # Handler of Console
    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.DEBUG)

    # Set Formatter
    formatter = logging.Formatter(f"%(asctime)s: %(message)s")
    handler_file.setFormatter(formatter)
    handler_console.setFormatter(formatter)

    # Add Handlers
    _logger.addHandler(handler_file)
    _logger.addHandler(handler_console)

@valid_init
def log_single_batch_info(
        epoch: int,
        total_epoch: int,
        batch: int,
        total_batch: int,
        loss: float,
        dice: float,
        time_cost: float,
        other_info:Optional[str] = None
) -> None:
    _logger.info(
        f"Epoch [{epoch+1:<4d}/{total_epoch:<4d}]"
        f"\tIter [{batch + 1:<4d}/{total_batch:<4d}]"
        f"\tLoss {loss:.6f}"
        f"\tDice {dice:.6f}"
        f"\tTime {time_cost:.6f}s/iter"
        + (other_info if other_info is not None else "")
    )

@valid_init
def log_valid_end_info(
        epoch: int,
        loss: float,
        dice: float
):
    # Epoch Validation Done
    _logger.info(F"Validation end at Epoch {epoch + 1} with {loss=} & {dice=}!")

@valid_init
def log_expected_time(
        time_per_train: float,
        time_per_valid: float,
        epoch_now: int, 
        epoch_total: int,
        valid_step: int
) -> None:
    epoch_left_train = epoch_total - epoch_now - 1
    epoch_left_valid = (epoch_total - epoch_now) // valid_step + 1
    time_left = time_per_train * epoch_left_train + time_per_valid * epoch_left_valid
    time_expected = time.time() + time_left
    end_timepoint = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_expected))
    _logger.info("Expected to end at: {}".format(end_timepoint))

@valid_init
def log_expected_time(
        time_per_train_valid: float,
        epoch_now: int,
        epoch_total: int
) -> None:
    epoch_left_train = epoch_total - epoch_now - 1
    time_left = time_per_train_valid * epoch_left_train
    time_expected = time.time() + time_left
    end_timepoint = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_expected))
    _logger.info("Expected to end at: {}".format(end_timepoint))

@valid_init
def log_info(info: str):
    _logger.info(info)