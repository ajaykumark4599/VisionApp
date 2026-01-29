from core.registry import BLOCKS

def main():
    print("\nVISION PIPELINE BUILDER (Ubuntu)")
    print("--------------------------------")

    print("Available Experiments:")
    for key, (name, _) in BLOCKS.items():
        print(f"{key}. {name}")

    choice = input("\nSelect experiment number: ").strip()

    if choice not in BLOCKS:
        print("Invalid choice")
        return

    input_path = input("Enter input image path: ").strip()
    output_dir = "data/output"

    params = None

    # Experiments needing extra inputs
    if choice in ["8", "9", "10", "14"]:
        second_image = input("Enter second image path: ").strip()
        params = {"second_image": second_image}

    if choice == "8":
        print("Enter 4 source points (x y):")
        src_points = [list(map(int, input().split())) for _ in range(4)]

        print("Enter 4 destination points (x y):")
        dst_points = [list(map(int, input().split())) for _ in range(4)]

        params = {
            "src_points": src_points,
            "dst_points": dst_points
        }

    _, run_func = BLOCKS[choice]
    result = run_func(input_path, output_dir, params)

    print("\nResult:")
    print(result)

if __name__ == "__main__":
    main()
