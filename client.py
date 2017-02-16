try:
    from MinimalServer import MinimalClient
    from server import W2VServer
except ImportError:
    from .MinimalServer import MinimalClient
    from .server import W2VServer


def get_w2v_client(host='localhost', port=7443):
    client = MinimalClient(W2VServer,
        host=host, port=port, buffersize=4096
    )
    return client
