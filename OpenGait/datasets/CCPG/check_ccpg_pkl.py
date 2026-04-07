import os
import pickle
import argparse
import numpy as np


def describe_array(name, arr):
    try:
        shape = arr.shape
        dtype = arr.dtype
        info = {
            "name": name,
            "type": type(arr).__name__,
            "shape": shape,
            "dtype": str(dtype),
        }
        if isinstance(arr, np.ndarray):
            info.update({
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
            })
        return info
    except Exception as e:
        return {"name": name, "error": str(e)}


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Inspect CCPG PKL outputs in a sample directory.")
    parser.add_argument("sample_dir", type=str, help="Directory like <output>/<id>/<type>/<view>")
    parser.add_argument("--print_first_frames", type=int, default=0, help="Print a small summary of first N frames per array")
    args = parser.parse_args()

    if not os.path.isdir(args.sample_dir):
        raise FileNotFoundError(f"Not a directory: {args.sample_dir}")

    targets = [
        "-rgbs.pkl",
        "-sils.pkl",
        "-aligned-sils.pkl",
        "-ratios.pkl",
    ]

    found = []
    for fn in sorted(os.listdir(args.sample_dir)):
        for suf in targets:
            if fn.endswith(suf):
                found.append(fn)
                break

    if not found:
        print("No CCPG PKL files found in the directory.")
        return

    print(f"Found {len(found)} PKL files:")
    for fn in found:
        p = os.path.join(args.sample_dir, fn)
        try:
            obj = load_pkl(p)
        except Exception as e:
            print(f"  {fn}: failed to load ({e})")
            continue

        # If dict, list keys; else describe array directly
        if isinstance(obj, dict):
            print(f"  {fn}: dict with keys {list(obj.keys())}")
            for k, v in obj.items():
                info = describe_array(f"{fn}:{k}", v)
                print(f"    - {info}")
        else:
            info = describe_array(fn, obj)
            print(f"  - {info}")

        # Optional sample preview
        if args.print_first_frames and isinstance(obj, np.ndarray):
            n = min(args.print_first_frames, len(obj)) if obj.ndim > 0 else 0
            if n > 0:
                print(f"    Previewing first {n} frame(s) stats:")
                for i in range(n):
                    arr = obj[i]
                    arr = np.asarray(arr)
                    print(f"      [{i}] shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.4f}")


if __name__ == "__main__":
    main()
