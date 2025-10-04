# prod_assistant/core/logger.py

import os
import logging
from datetime import datetime
import structlog
from prod_assistant.core.trace import get_trace_id


class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def add_trace_id(self, logger, method_name, event_dict):
        """Structlog processor that injects the current trace_id into logs."""
        event_dict["trace_id"] = get_trace_id()
        return event_dict

    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name)

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[console_handler, file_handler]
        )

        # structlog.configure(
        #     processors=[
        #         structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        #         structlog.processors.add_log_level,
        #         structlog.processors.EventRenamer(to="event"),
        #         structlog.processors.JSONRenderer()
        #     ],
        #     logger_factory=structlog.stdlib.LoggerFactory(),
        #     cache_logger_on_first_use=True,
        # )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                self.add_trace_id,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            wrapper_class=structlog.make_filtering_bound_logger(20),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(logger_name)
