import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.datasets import get_loaders
from src.models import create_mobilenet, create_resnet18, SmallCNN

def find_run_dirs(base="results", model_name="mobilenetv2", run_tag: str = "") -> List[str]:
    hits = []
    if not os.path.isdir(base):
        return hits
    for root, dirs, files in os.walk(base):
        if f"{model_name}_best.pt" in files or "best_model.pt" in files:
            # chấp nhận cả kiểu tên cũ (best_model.pt) lẫn mới (<model>_best.pt)
            if model_name in root:
                if (run_tag.lower() in root.lower()) if run_tag else True:
                    hits.append(root)
    return hits

def pick_latest_run(run_dirs: List[str]) -> Optional[str]:
    if not run_dirs:
        return None
    return max(run_dirs, key=os.path.getmtime)

def guess_model_from_ckpt(ckpt_path: str):
    name = os.path.basename(ckpt_path).lower()
    if "mobilenetv2" in name:
        return "mobilenetv2"
    if "resnet18" in name:
        return "resnet18"
    if "smallcnn" in name or "best_model" in name:
        return "smallcnn"
    # fallback
    return "smallcnn"

def load_state_and_classes(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("classes", None)
    return ckpt, None

def build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "mobilenetv2":
        return create_mobilenet(num_classes=num_classes, freeze=False, pretrained=False)
    if arch == "resnet18":
        return create_resnet18(num_classes=num_classes, freeze=False, pretrained=False)
    return SmallCNN(num_classes=num_classes)

def evaluate_one(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy()
            y_pred.append(p); y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/intel")
    ap.add_argument("--models", nargs="*", default=["smallcnn", "mobilenetv2", "resnet18"],
                    help="Danh sách mô hình cần chấm: smallcnn mobilenetv2 resnet18")
    ap.add_argument("--run_tag", type=str, default="",
                    help="Lọc theo tag của run (ví dụ: head_only, finetune_20e). Bỏ trống để lấy run mới nhất.")
    ap.add_argument("--img_size", type=int, default=160)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load test loader (chia sẻ cho mọi mô hình)
    _, _, test_loader, classes = get_loaders(args.data_root, img_size=args.img_size, batch_size=64, num_workers=2)
    num_classes = len(classes)

    # Tập hợp summary
    summary_rows = [("model", "run_dir", "test_acc")]

    for model_name in args.models:
        # 1) Tìm run phù hợp
        run_dirs = find_run_dirs(base="results", model_name=model_name, run_tag=args.run_tag)
        run_dir = pick_latest_run(run_dirs)
        if not run_dir:
            print(f"[WARN] Không tìm thấy run cho mô hình '{model_name}' (tag='{args.run_tag}'). Bỏ qua.")
            continue

        # 2) Chọn checkpoint
        ckpt_candidates = [
            os.path.join(run_dir, f"{model_name}_best.pt"),
            os.path.join(run_dir, "best_model.pt"),  # tương thích phiên bản cũ
        ]
        ckpt_path = next((p for p in ckpt_candidates if os.path.isfile(p)), None)
        if not ckpt_path:
            print(f"[WARN] Không có checkpoint trong {run_dir}. Bỏ qua.")
            continue

        print(f"\n=== Evaluating {model_name} ===")
        print("Run dir :", run_dir)
        print("Checkpoint:", ckpt_path)

        # 3) Load state_dict + classes
        state_dict, classes_from_ckpt = load_state_and_classes(ckpt_path, device)
        model_arch = guess_model_from_ckpt(ckpt_path)
        # nếu ckpt có classes thì ưu tiên
        eval_classes = classes_from_ckpt if classes_from_ckpt else classes
        n_cls = len(eval_classes)

        # 4) Build & load model
        model = build_model(model_arch, num_classes=n_cls).to(device)
        model.load_state_dict(state_dict)

        # 5) Đánh giá
        y_true, y_pred = evaluate_one(model, test_loader, device)
        acc = (y_true == y_pred).mean()

        # 6) Lưu classification report + confusion matrix vào đúng run dir
        ensure_dir(run_dir)
        report_txt = os.path.join(run_dir, "classification_report.txt")
        cm_png = os.path.join(run_dir, "confusion_matrix.png")

        report = classification_report(y_true, y_pred, target_names=eval_classes, digits=3)
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=eval_classes, xticks_rotation=45, cmap="Blues", normalize=None
        )
        plt.tight_layout()
        plt.savefig(cm_png, dpi=200)
        plt.close()
        print(f"Saved: {cm_png} and {report_txt}")
        summary_rows.append((model_name, run_dir, f"{acc:.4f}"))

    # 7) Ghi summary CSV
    if len(summary_rows) > 1:
        import csv
        summary_csv = os.path.join("results", "summary_eval.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(summary_rows)
        print(f"\n✅ Summary saved: {summary_csv}")
        for r in summary_rows[1:]:
            print(" -", r[0], "|", r[1], "| test_acc:", r[2])
    else:
        print("\n[INFO] Không có mô hình nào được chấm điểm.")

if __name__ == "__main__":
    main()
