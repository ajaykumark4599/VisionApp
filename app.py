from core.pipeline import Pipeline
from core.registry import BLOCKS
import os

print("VISION PIPELINE BUILDER (Ubuntu)")
print("--------------------------------")

pipeline = Pipeline()
input_image = "data/input/test.jpg"
output_dir = "data/output"

use_saved = input("Load saved pipeline? (y/n): ").strip().lower()

if use_saved == "y" and os.path.exists("pipeline.txt"):
    print("Loading pipeline from pipeline.txt")
    with open("pipeline.txt", "r") as f:
        choices = [line.strip() for line in f.readlines()]
else:
    print("Available Blocks:")
    for key, (name, _) in BLOCKS.items():
        print(f"{key}. {name}")

    user_input = input("Enter block numbers (comma separated, e.g. 1,2): ").strip()
    choices = [c.strip() for c in user_input.split(",")]

    # Save pipeline
    with open("pipeline.txt", "w") as f:
        for c in choices:
            f.write(c + "\n")

    print("Pipeline saved to pipeline.txt")

for choice in choices:
    if choice in BLOCKS:
        name, block_func = BLOCKS[choice]
        params = {}

        if name == "Edge Detection":
            low = input("Enter low threshold (default 100): ").strip()
            high = input("Enter high threshold (default 200): ").strip()

            if low:
                params["low"] = int(low)
            if high:
                params["high"] = int(high)

        pipeline.add(block_func, params)
    else:
        print(f"Invalid choice: {choice}")

result = pipeline.run(input_image, output_dir)

print("Pipeline finished")
print(result)

