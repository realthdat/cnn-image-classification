# scripts/plot_curves.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# === 1️⃣ Tùy chỉnh model cần vẽ ===
MODEL_NAME = "resnet18"   # có thể đổi: smallcnn, mobilenetv2, resnet18
RUN_TAG = ""                 # để trống sẽ tự chọn run gần nhất, hoặc ghi "baseline", "head_only", v.v.

# === 2️⃣ Xác định thư mục logs ===
base_dir = os.path.join("results")
sub_dirs = []

# tìm tất cả run thuộc mô hình này
for root, dirs, files in os.walk(base_dir):
    if "logs.csv" in files and MODEL_NAME in root:
        sub_dirs.append(root)

if not sub_dirs:
    raise FileNotFoundError(f"Không tìm thấy logs.csv cho mô hình {MODEL_NAME} trong thư mục results/")

# nếu có nhiều run, chọn theo RUN_TAG hoặc mới nhất
if RUN_TAG:
    candidates = [d for d in sub_dirs if RUN_TAG.lower() in d.lower()]
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy run có tag '{RUN_TAG}'")
    run_dir = candidates[0]
else:
    run_dir = max(sub_dirs, key=os.path.getmtime)

print(f"Đang đọc logs từ: {run_dir}")
df = pd.read_csv(os.path.join(run_dir, "logs.csv"))

# === 3️⃣ Vẽ Accuracy curve ===
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["tr_acc"], label="Train")
plt.plot(df["epoch"], df["val_acc"], label="Val")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title(f"{MODEL_NAME} – Accuracy vs. Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "acc_curve.png"), dpi=200)

# === 4️⃣ Vẽ Loss curve ===
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["tr_loss"], label="Train")
plt.plot(df["epoch"], df["val_loss"], label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title(f"{MODEL_NAME} – Loss vs. Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=200)

print(f"✅ Saved plots in: {run_dir}")
print("Files:")
print(" - acc_curve.png")
print(" - loss_curve.png")
