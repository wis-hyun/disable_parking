from pathlib import Path

BASE_DIR = Path("disabled_sign_detection")
SPLITS = ["train", "valid", "test"]  

for split in SPLITS:
    label_dir = BASE_DIR / split / "labels"
    if not label_dir.exists():
        print(f"❌ {label_dir} not found")
        continue

    for txt in label_dir.glob("*.txt"):
        lines = txt.read_text().splitlines()
        new_lines = []

        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            parts[0] = "0"   # 모든 class id를 0으로 통일
            new_lines.append(" ".join(parts))

        txt.write_text("\n".join(new_lines))

print("✅ All labels converted to single class (0: disabled_sign)")