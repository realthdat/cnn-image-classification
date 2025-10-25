# scripts/compare_models.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import torch
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import accuracy_score

from src.datasets import get_loaders
from src.models import create_mobilenet, create_resnet18, SmallCNN

# ---------- Helpers ----------
def find_run_dirs(base="results", model_name="mobilenetv2", run_tag: str = "") -> List[str]:
    hits = []
    if not os.path.isdir(base):
        return hits
    for root, _, files in os.walk(base):
        if "logs.csv" in files and model_name in root:
            if (run_tag.lower() in root.lower()) if run_tag else True:
                hits.append(root)
    return hits

def pick_latest_run(run_dirs: List[str]) -> Optional[str]:
    if not run_dirs:
        return None
    return max(run_dirs, key=os.path.getmtime)

def read_best_val_acc(run_dir: str) -> Optional[float]:
    log_path = os.path.join(run_dir, "logs.csv")
    if not os.path.isfile(log_path):
        return None
    df = pd.read_csv(log_path)
    if "val_acc" not in df.columns:
        return None
    return float(df["val_acc"].max())

def guess_ckpt_path(run_dir: str, model_name: str) -> Optional[str]:
    # ưu tiên <model>_best.pt, fallback best_model.pt
    c1 = os.path.join(run_dir, f"{model_name}_best.pt")
    c2 = os.path.join(run_dir, "best_model.pt")
    if os.path.isfile(c1): return c1
    if os.path.isfile(c2): return c2
    return None

def guess_model_from_name(model_name: str):
    m = model_name.lower()
    if m == "mobilenetv2": return "mobilenetv2"
    if m == "resnet18": return "resnet18"
    return "smallcnn"

def build_model(arch: str, num_classes: int):
    if arch == "mobilenetv2":
        return create_mobilenet(num_classes=num_classes, freeze=False, pretrained=False)
    if arch == "resnet18":
        return create_resnet18(num_classes=num_classes, freeze=False, pretrained=False)
    return SmallCNN(num_classes=num_classes)

def load_state_and_classes(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("classes", None)
    return ckpt, None

def compute_test_acc(ckpt_path: str, arch: str, data_root: str, img_size: int, batch_size: int = 64) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, classes = get_loaders(data_root, img_size=img_size, batch_size=batch_size, num_workers=2)
    n_cls = len(classes)
    state_dict, classes_from_ckpt = load_state_and_classes(ckpt_path, device)
    if classes_from_ckpt:  # nếu ckpt có classes thì tin theo ckpt
        n_cls = len(classes_from_ckpt)
    model = build_model(arch, num_classes=n_cls).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy()
            y_pred.append(p); y_true.append(y.numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    return float(accuracy_score(y_true, y_pred))

def maybe_read_summary_test_acc(summary_csv: str, model_name: str, run_dir: str) -> Optional[float]:
    if not os.path.isfile(summary_csv):
        return None
    try:
        df = pd.read_csv(summary_csv, header=None, names=["model","run_dir","tr_acc"])
        df = df[df["model"].str.lower() == model_name.lower()]
        # nếu có nhiều dòng, ưu tiên dòng có run_dir trùng
        if run_dir:
            sel = df[df["run_dir"] == run_dir]
            if not sel.empty:
                return float(sel.iloc[-1]["tr_acc"])
        if not df.empty:
            return float(df.iloc[-1]["tr_acc"])
        return None
    except Exception:
        return None

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/intel")
    ap.add_argument("--models", nargs="*", default=["smallcnn","mobilenetv2","resnet18"])
    ap.add_argument("--run_tag", type=str, default="", help="lọc theo tag thư mục run; để trống lấy run mới nhất")
    ap.add_argument("--img_size", type=int, default=160)
    args = ap.parse_args()

    results = []  # (model, run_dir, best_val_acc, test_acc)

    for model_name in args.models:
        run_dirs = find_run_dirs("results", model_name, run_tag=args.run_tag)
        run_dir = pick_latest_run(run_dirs)
        if not run_dir:
            print(f"[WARN] Không tìm thấy run cho {model_name} (tag='{args.run_tag}'). Bỏ qua.")
            continue

        best_val = read_best_val_acc(run_dir)
        if best_val is None:
            print(f"[WARN] {model_name}: thiếu logs.csv hoặc cột val_acc trong {run_dir}.")
            continue

        # test_acc: ưu tiên đọc từ summary_eval.csv do src.eval ghi
        summary_csv = os.path.join("results", "summary_eval.csv")
        test_acc = maybe_read_summary_test_acc(summary_csv, model_name, run_dir)

        # fallback: tính test acc trực tiếp
        if test_acc is None:
            ckpt_path = guess_ckpt_path(run_dir, model_name)
            if ckpt_path is None:
                print(f"[WARN] {model_name}: không tìm thấy checkpoint trong {run_dir}.")
                continue
            arch = guess_model_from_name(model_name)
            test_acc = compute_test_acc(ckpt_path, arch, data_root=args.data_root, img_size=args.img_size)

        results.append((model_name, run_dir, best_val, test_acc))

    if not results:
        print("[INFO] Không có kết quả để vẽ.")
        return

    # --- Plot: Best Val Acc ---
    models = [r[0] for r in results]
    val_accs = [r[2] for r in results]
    plt.figure(figsize=(7,4))
    plt.bar(models, val_accs)
    plt.ylim(0, 1.0)
    plt.ylabel("Best Val Accuracy")
    plt.title("So sánh Best Val Acc giữa các mô hình")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/compare_val_acc.png", dpi=200)
    plt.close()

    # --- Plot: Test Acc ---
    test_accs = [r[3] for r in results]
    plt.figure(figsize=(7,4))
    plt.bar(models, test_accs)
    plt.ylim(0, 1.0)
    plt.ylabel("Test Accuracy")
    plt.title("So sánh Test Acc giữa các mô hình")
    plt.tight_layout()
    plt.savefig("results/compare_test_acc.png", dpi=200)
    plt.close()

    # In tóm tắt
    print("✅ Đã lưu:")
    print(" - results/compare_val_acc.png")
    print(" - results/compare_test_acc.png")
    print("\nBảng tóm tắt:")
    for m, rd, va, ta in results:
        print(f"  {m:12s} | val_best={va:.4f} | test={ta:.4f} | {rd}")

if __name__ == "__main__":
    main()