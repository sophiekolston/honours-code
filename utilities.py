import time


# timer to call on code to count time of execution
#   don't think it is that accurate
class Timer:
    def __init__(self, code_name='function'):
        self.start_time = None
        self.name = code_name

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        self.start_time = None
        
        print(f'Executed {self.name} in: {elapsed_time:0.2f} s')
