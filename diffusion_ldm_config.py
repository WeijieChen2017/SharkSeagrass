global_config = {}

def set_param(key, value):
    global_config[key] = value

def get_param(key, default=None):
    if key in global_config.keys():
        return global_config[key]
    return default