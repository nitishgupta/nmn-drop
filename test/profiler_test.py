import time
import collections
import random


def profile_func_decorator(input_func):
    def timed_func(*args, **kwargs):
        with Profiler(input_func.__name__):
            result = input_func(*args, **kwargs)
        return result
    return timed_func


class Model():
    def __init__(self):
        print("Init class")

    @profile_func_decorator
    def __call__(self):
        randnum = random.random()
        if randnum > 0.6:
            with Profiler(scope_name="scope_1"):
                time.sleep(1)
        else:
            with Profiler(scope_name="scope_2"):
                time.sleep(0.5)

        self.my_func(random.random())

    @profile_func_decorator
    def my_func(self, x):
        time.sleep(0.3)
        return x + 1



class Profiler:
    timer_dict = collections.defaultdict(float)
    num_calls = collections.defaultdict(int)
    def __init__(self, scope_name: str):
        self.scope_name = scope_name
        self.start_time = None
        self.end_time = None
        self.total_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        Profiler.timer_dict[self.scope_name] += self.total_time
        Profiler.num_calls[self.scope_name] += 1

    @staticmethod
    def to_string():
        s = "------------------------  Profiler Stats  ------------------------\n"
        s += "Scope \t Num_Calls \t TimeElapsed\n"
        for k, v in Profiler.timer_dict.items():
            num_calls = Profiler.num_calls[k]
            s += "{} \t {} \t {} seconds\n".format(k, num_calls, v)
        s += "----------------------------------------------------------------"
        return s





if __name__ == "__main__":
    m = Model()
    for i in range(10):
        m()

    print(Profiler.to_string())



