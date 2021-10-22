import warnings
with warnings.catch_warnings():
    # XXX: websocket-client library seems to have leaks on connection
    #      retry that cause annoying warnings within Python 3.8+
    # warnings.filterwarnings("ignore", category=ResourceWarning)
    print(warnings.catch_warnings())