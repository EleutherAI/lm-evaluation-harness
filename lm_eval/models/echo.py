from array import array
import sys


def echo():
    from .byte import SubprocessLM

    return SubprocessLM.start(["python", __file__])


def _echo_loglikelihood():
    while True:
        b = sys.stdin.buffer.read(1)
        if len(b) == 0:
            break
        o = array("f", [-float("inf")] * 256)
        o[b[0]] = 1.0
        sys.stdout.buffer.write(o)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    _echo_loglikelihood()
