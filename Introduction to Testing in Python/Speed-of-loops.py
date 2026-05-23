def test_list(benchmark):
	# Add decorator here
    @benchmark
    def iterate_list():
		# Complete the loop here
        for el in [i for i in range(1000)]:
            pass

def test_set(benchmark):
	# Add decorator here
    @benchmark
    def iterate_set():
        # Complete the loop here
        for el in {i for i in range(1000)}:
            pass