import os
import argparse
import pickle
import numpy as np
import cv2


def _ensure_chw_rgb(arr: np.ndarray) -> np.ndarray:
	if arr.ndim != 4:
		raise ValueError(f"rgbs ndim must be 4, got {arr.ndim}")
	if arr.shape[1] == 3:
		return arr
	if arr.shape[-1] == 3:
		return np.transpose(arr, (0, 3, 1, 2))
	raise ValueError(f"rgbs must be CHW or HWC with 3 channels, got shape {arr.shape}")


def _ensure_masks(arr: np.ndarray, n: int, h: int, w: int) -> np.ndarray:
	if arr.ndim == 4 and arr.shape[1] == 1:
		arr = arr[:, 0]
	if arr.ndim != 3:
		raise ValueError(f"sils ndim must be 3 (N,H,W), got {arr.ndim}")
	if arr.shape[0] != n:
		raise ValueError(f"sils N mismatch: {arr.shape[0]} vs {n}")
	if arr.shape[1] != h or arr.shape[2] != w:
		raise ValueError(f"sils size mismatch: {(arr.shape[1], arr.shape[2])} vs {(h, w)}")
	return arr


def main():
	parser = argparse.ArgumentParser(description="Multiply RGBs with silhouettes to get foreground and save a sample PNG.")
	parser.add_argument("--rgbs", required=True, help="Path to *-rgbs.pkl")
	parser.add_argument("--sils", required=True, help="Path to *-sils.pkl")
	parser.add_argument("--out_png", default=None, help="Output PNG path for a sample foreground frame")
	parser.add_argument("--save_fg_pkl", action="store_true", help="Also save full foreground PKL next to rgbs.pkl")
	parser.add_argument("--fg_pkl", default=None, help="Optional custom path for foreground PKL output")
	args = parser.parse_args()

	with open(args.rgbs, "rb") as f:
		rgbs = pickle.load(f)
	with open(args.sils, "rb") as f:
		sils = pickle.load(f)

	rgbs = _ensure_chw_rgb(np.asarray(rgbs))
	n, c, h, w = rgbs.shape
	sils = _ensure_masks(np.asarray(sils), n, h, w)

	masks = (sils > 0).astype(rgbs.dtype)  # (N,H,W)
	masks = masks[:, None, :, :]           # (N,1,H,W)
	fg = rgbs * masks                      # (N,3,H,W)

	# Save sample PNG
	if n > 0:
		out_dir = os.path.dirname(args.rgbs)
		out_png = args.out_png or os.path.join(out_dir, "fg-sample.png")
		sample = fg[0]
		hwc = np.transpose(sample, (1, 2, 0))
		bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
		cv2.imwrite(out_png, bgr)
		print(f"Saved sample PNG: {out_png}")

	# Optionally save foreground PKL
	if args.save_fg_pkl:
		if args.fg_pkl is not None:
			fg_path = args.fg_pkl
		else:
			base, name = os.path.split(args.rgbs)
			if name.endswith("-rgbs.pkl"):
				name = name.replace("-rgbs.pkl", "-fg-rgbs.pkl")
			else:
				name = f"fg-{name}"
			fg_path = os.path.join(base, name)
		with open(fg_path, "wb") as f:
			pickle.dump(fg, f)
		print(f"Saved foreground PKL: {fg_path}")

	print("Done. Shapes:")
	print("  rgbs:", rgbs.shape, rgbs.dtype)
	print("  sils:", sils.shape, sils.dtype)
	print("  fg:  ", fg.shape, fg.dtype)


if __name__ == "__main__":
	main()

