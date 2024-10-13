def grid(*args):
    if len(args) == 1:
        for k in args[0]:
            yield [k]
    else:
        for k in args[0]:
            for rest in grid(*args[1:]):
                yield([k] + rest)