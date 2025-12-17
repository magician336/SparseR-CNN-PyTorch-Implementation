import argparse
import csv
import os
from collections import OrderedDict


def _read_loss_csv(csv_path: str):
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	rows = []
	with open(csv_path, "r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for r in reader:
			if not r:
				continue
			rows.append(r)
	if not rows:
		raise ValueError(f"CSV is empty: {csv_path}")
	return rows


def _to_int(x, default=None):
	try:
		return int(float(x))
	except Exception:
		return default


def _to_float(x, default=None):
	try:
		return float(x)
	except Exception:
		return default


def _pick_epoch_last_rows(rows):
	"""Pick the last row of each epoch.

	Preference order:
	1) rows where iter == iter_total (end of epoch)
	2) otherwise, max iter within the epoch
	"""
	by_epoch = {}
	for r in rows:
		e = _to_int(r.get("epoch"), None)
		if e is None:
			continue
		it = _to_int(r.get("iter"), -1)
		it_total = _to_int(r.get("iter_total"), None)
		is_end = (it_total is not None) and (it == it_total)
		cur = by_epoch.get(e)
		if cur is None:
			by_epoch[e] = (is_end, it, r)
		else:
			cur_is_end, cur_it, _ = cur
			if is_end and not cur_is_end:
				by_epoch[e] = (is_end, it, r)
			elif (is_end == cur_is_end) and (it >= cur_it):
				by_epoch[e] = (is_end, it, r)

	# sort by epoch
	picked = OrderedDict()
	for e in sorted(by_epoch.keys()):
		picked[e] = by_epoch[e][2]
	if not picked:
		raise ValueError("No valid epoch rows found in CSV.")
	return picked


def plot_losses(csv_path: str, out_path: str, title: str = "Loss Curves"):
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as e:
		raise RuntimeError(
			"matplotlib is required for plotting. Install it with: pip install matplotlib"
		) from e

	rows = _read_loss_csv(csv_path)
	epoch_rows = _pick_epoch_last_rows(rows)

	keys = ["avg_total_loss", "avg_loss_bbox", "avg_loss_ce", "avg_loss_giou"]
	missing = [k for k in keys if k not in next(iter(epoch_rows.values())).keys()]
	if missing:
		raise KeyError(f"Missing columns in CSV: {missing}. Available: {list(next(iter(epoch_rows.values())).keys())}")

	epochs = list(epoch_rows.keys())
	series = {k: [] for k in keys}
	for e, r in epoch_rows.items():
		for k in keys:
			v = _to_float(r.get(k), None)
			series[k].append(v)

	plt.figure(figsize=(10, 6))
	for k in keys:
		plt.plot(epochs, series[k], label=k)
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.title(title)
	plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
	plt.legend()

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Plot loss curves from loss_log.csv")
	parser.add_argument("--csv", required=True, help="Path to loss_log.csv")
	parser.add_argument("--out", required=True, help="Output image path (e.g., loss.png)")
	parser.add_argument("--title", default="Loss Curves", help="Figure title")
	args = parser.parse_args()

	plot_losses(args.csv, args.out, title=args.title)
	print(f"Saved: {args.out}")


if __name__ == "__main__":
	main()
