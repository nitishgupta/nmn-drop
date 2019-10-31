import time
import collections
from utils import util

class Profile:
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
        Profile.timer_dict[self.scope_name] += self.total_time
        Profile.num_calls[self.scope_name] += 1

    @staticmethod
    def to_string():
        perc_of_forward = collections.defaultdict(float)
        if "forward" in Profile.timer_dict:
            forward_time = Profile.timer_dict["forward"]
            for k, v in Profile.timer_dict.items():
                perc = (v/forward_time) * 100.0
                perc_of_forward[k] = perc

        timer_dict = util.round_all(Profile.timer_dict, prec=4)
        perc_of_forward = util.round_all(perc_of_forward, prec=4)

        s = "\n------------------------  Profiler Stats  ------------------------\n"
        s += "Scope \t Num_Calls \t TimeElapsed \t Perc_of_Forward\n"
        for k, v in timer_dict.items():
            num_calls = Profile.num_calls[k]
            perc_forward = perc_of_forward[k] if k in perc_of_forward else 0.0
            s += "{} \t {} \t {} seconds \t {} % \n".format(k, num_calls, v, perc_forward)
        s += "----------------------------------------------------------------\n"
        return s


def profile_func_decorator(input_func):
    def timed_func(*args, **kwargs):
        with Profile(input_func.__name__):
            result = input_func(*args, **kwargs)
        return result
    return timed_func
