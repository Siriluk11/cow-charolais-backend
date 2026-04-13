import os
import cv2 # type: ignore

def augment_images(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path)

        if img is None:
            continue

        flip = cv2.flip(img, 1)
        rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        name, ext = os.path.splitext(filename)

        cv2.imwrite(os.path.join(folder_path, name + "_flip" + ext), flip)
        cv2.imwrite(os.path.join(folder_path, name + "_rot" + ext), rotate)

base_path = "dataset"

for class_name in ['1','2','3','4','5']:
    augment_images(os.path.join(base_path, class_name))

print("Augmentation เสร็จแล้ว")