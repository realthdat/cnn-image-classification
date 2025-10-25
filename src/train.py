import os, csv, argparse, datetime as dt
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

from src.datasets import get_loaders
from src.models import SmallCNN, create_mobilenet, create_resnet18
from src.utils import set_seed

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def build_model(name, num_classes, freeze, pretrained=True):
    name = name.lower()
    if name == "smallcnn":
        return SmallCNN(num_classes)
    if name == "mobilenetv2":
        return create_mobilenet(num_classes, freeze=freeze, pretrained=pretrained)
    if name == "resnet18":
        return create_resnet18(num_classes, freeze=freeze, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")

def make_run_dir(base="results", model="mobilenetv2", freeze=False, tag=None):
    # ví dụ: results/mobilenetv2_freeze/2025-10-25_21-08-03  (hoặc thêm tag)
    model_folder = f"{model}{'_freeze' if freeze else ''}"
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    leaf = tag.strip().replace(" ", "_") if tag else stamp
    out_dir = os.path.join(base, model_folder, leaf)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/intel")
    ap.add_argument("--model", type=str, default="mobilenetv2",
                    choices=["smallcnn", "mobilenetv2", "resnet18"])
    ap.add_argument("--freeze", action="store_true",
                    help="freeze backbone (head-only) đối với các model pretrained")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="không dùng pretrained ImageNet cho TL models")
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="",
                    help="chuỗi gắn nhãn run, ví dụ: --tag ig224_ft_lr1e-4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # chuẩn bị loaders theo img_size/batch_size
    from torchvision import transforms
    tfm_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    # get_loaders dùng mặc định img_size=160; ở đây ta tạm thay đổi transform bằng cách monkey-patch
    # hoặc bạn có thể sửa get_loaders để nhận tham số. Đơn giản nhất: tạo loaders thủ công ở đây.
    # Ta dùng get_loaders hiện có để lấy classes rồi thay DataLoader:
    train_loader, val_loader, test_loader, classes = get_loaders(args.data_root, img_size=args.img_size,
                                                                 batch_size=args.batch_size, num_workers=args.workers)
    num_classes = len(classes)
    print("Classes:", classes)

    model = build_model(args.model, num_classes, freeze=args.freeze, pretrained=not args.no_pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.wd)

    # tạo thư mục run riêng
    out_dir = make_run_dir(base="results", model=args.model, freeze=args.freeze, tag=args.tag)
    csv_path = os.path.join(out_dir, "logs.csv")
    ckpt_path = os.path.join(out_dir, f"{args.model}_best.pt")

    # ghi config để dễ truy vết
    with open(os.path.join(out_dir, "hparams.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"model={args.model}\nfreeze={args.freeze}\npretrained={not args.no_pretrained}\n"
            f"img_size={args.img_size}\nbatch_size={args.batch_size}\nepochs={args.epochs}\n"
            f"lr={args.lr}\nwd={args.wd}\nworkers={args.workers}\nseed={args.seed}\n"
            f"data_root={args.data_root}\nclasses={classes}\n"
        )

    # CSV header
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["epoch", "tr_loss", "tr_acc", "val_loss", "val_acc"])

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"
            ])

        print(f"[{epoch:02d}] TrainAcc {tr_acc*100:.2f}% | ValAcc {val_acc*100:.2f}% | "
              f"TrainLoss {tr_loss:.4f} | ValLoss {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "val_acc": float(best_acc)}, ckpt_path)
            print(f"  → Saved {ckpt_path} (best acc {best_acc*100:.2f}%)")

    print(f"Done. Best ValAcc = {best_acc*100:.2f}%")
    print(f"Artifacts saved in: {out_dir}")

if __name__ == "__main__":
    main()
