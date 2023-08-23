import logging

import structlog

# default configuration + whatever adjustments i list here:
# * pad_event: padding for event defaults to 30. i prefer smaller.
ped_event = 12

# console_renderer = structlog.dev.ConsoleRenderer(pad_event=ped_event)
# console_renderer._longest_level
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.dev.ConsoleRenderer(pad_event=ped_event),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

get_logger = structlog.get_logger
