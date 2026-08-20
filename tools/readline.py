"""Dummy readline module for Windows compatibility with panstamps."""
# Provide minimal interface that panstamps.cl_utils expects
def parse_and_bind(*args, **kwargs): pass
def read_history_file(*args, **kwargs): pass
def write_history_file(*args, **kwargs): pass
def set_completer(*args, **kwargs): pass
def set_history_length(*args, **kwargs): pass
def get_history_length(): return 100
