import urllib.parse

class FieldStorage:
    def __init__(self, fp=None, environ=None, keep_blank_values=False, strict_parsing=False):
        pass

def parse_header(header):
    return header.split(";")[0].strip()

class MiniFieldStorage:
    def __init__(self, name, value):
        self.name = name
        self.value = value
