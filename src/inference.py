import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from typing import List

from src.models import create_mobilenet, create_resnet18, SmallCNN

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def build_model_from_ckpt(ckpt_path: str, num_classes: int):
    """Đoán kiến trúc từ tên file checkpoint, rồi khởi tạo đúng model."""
    name = os.path.basename(ckpt_path).lower()
    if "mobilenetv2" in name:
        return create_mobilenet(num_classes=num_classes, freeze=False, pretrained=False)
    if "resnet18" in name:
        return create_resnet18(num_classes=num_classes, freeze=False, pretrained=False)
    # fallback: SmallCNN
    return SmallCNN(num_classes=num_classes)

def load_checkpoint(model_path: str, device: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        classes = ckpt.get("classes", None)
    else:
        # Trường hợp lưu state_dict thuần
        state_dict = ckpt
        classes = None
    return state_dict, classes

def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

@torch.no_grad()
def infer_folder(folder: str, model_path: str, img_size: int = 160, topk: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = make_transform(img_size)

    # Load ckpt
    state_dict, classes = load_checkpoint(model_path, device)
    if classes is None or not isinstance(classes, (list, tuple)) or len(classes) == 0:
        # dự phòng: nếu ckpt không có classes, cố gắng lấy từ cấu trúc thư mục intel
        # (thứ tự theo ImageFolder)
        default_classes = ['buildings','forest','glacier','mountain','sea','street']
        classes = default_classes
    num_classes = len(classes)

    # Build model theo tên file ckpt
    model = build_model_from_ckpt(model_path, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Liệt kê ảnh
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Thư mục ảnh không tồn tại: {folder}")
    files: List[str] = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    if not files:
        print(f"Không tìm thấy ảnh ({IMG_EXTS}) trong: {folder}")
        return

    print(f"Device: {device} | Model: {os.path.basename(model_path)}")
    for img_name in sorted(files):
        path = os.path.join(folder, img_name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] {img_name}: lỗi đọc ảnh ({e})")
            continue

        x = transform(img).unsqueeze(0).to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        topv, topi = prob.topk(min(topk, num_classes))
        # in top-1 + top-k
        top1_idx = topi[0].item()
        top1_name = classes[top1_idx] if 0 <= top1_idx < num_classes else str(top1_idx)
        print(f"{img_name} → {top1_name} ({topv[0].item():.3f}) | top{topk}: " +
              ", ".join([f"{classes[i]}:{topv[j].item():.3f}" for j, i in enumerate(topi.cpu().tolist())]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="data/intel/seg_pred",
                    help="Thư mục chứa ảnh để dự đoán")
    ap.add_argument("--model_path", type=str, default="results/best_model.pt",
                    help="Đường dẫn tới checkpoint .pt đã train")
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    infer_folder(folder=args.folder, model_path=args.model_path,
                 img_size=args.img_size, topk=args.topk)

if __name__ == "__main__":
    main()
