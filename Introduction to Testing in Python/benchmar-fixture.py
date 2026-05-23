import time 
def test_funct(benchmark):
    benchmark(time.sleep, 1)

def test_funct1(benchmark):
    @benchmark
    def sleep_for_1_sec():
        time.sleep(1)