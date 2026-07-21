EMBEDDING_REGITSTRY = {}

def register(name):
    def wrapper(cls):
        EMBEDDING_REGITSTRY[name] = cls
        return cls
    return wrapper