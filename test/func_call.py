import sys


def func(a, b, **kwargs):
    print(a)
    print(b)
    print(kwargs)


func(a=5, c=10, d=5, b=6)


