import time

class Pipeline:
    def __init__(self):
        self.blocks = []

    def add(self, block_function, params=None):
        self.blocks.append((block_function, params or {}))

    def run(self, input_data, output_dir):
        results = []
        start_time = time.time()

        for block, params in self.blocks:
            result = block(input_data, output_dir, params)
            results.append(result)

        end_time = time.time()

        return {
            "results": results,
            "execution_time_sec": round(end_time - start_time, 3)
        }
