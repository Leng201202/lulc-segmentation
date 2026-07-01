from pathlib import Path
import cv2


def main():
    project_root = Path(__file__).resolve().parent
    test_image_path = project_root / "data/IRSAMap/train/image/1252.png"
    
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
        print(f"Original image shape (H, W, C): {img.shape}")
        print(f"Height: {img.shape[0]}")
        print(f"Width: {img.shape[1]}")
    else:
        print("Test image not found")


if __name__ == "__main__":
    main()
