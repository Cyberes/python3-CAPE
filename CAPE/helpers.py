def serialize_int64(obj):
    # Fix TypeError: Object of type int64 is not JSON serializable
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Type %s is not serializable" % type(obj))
