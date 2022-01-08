import logging


def initialize_logger():
    formatter = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-1s %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.setLevel(logging.INFO)
