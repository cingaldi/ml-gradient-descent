from time import perf_counter 

def timed(f):
    def _timed(*args, **kwargs):
        start = perf_counter()
        result = f(*args, **kwargs)
        finish = perf_counter()

        elapsed = finish - start

        #please carmine refactor me
        if elapsed < 1:
            timeValue , timeUnit = (int(elapsed * 1000) , "ms")
        else:
            timeValue , timeUnit = (int(elapsed) , "s")

        print("function {} executed in {}{}".format(f.__name__ , timeValue , timeUnit))

        return result
    return _timed