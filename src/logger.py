import logging


def initialize_logger():
    root = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-1s %(message)s')
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    root.setLevel(logging.INFO)