import cv2
import os
import numpy as np
import random

def simulate_tampering(image):
    img = image.copy()
    h, w = img.shape[:2]

    # Random circular region
    center = (
        random.randint(w // 4, 3 * w // 4),
        random.randint(h // 4, 3 * h // 4)
    )
    radius = random.randint(15, 40)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Brightness manipulation inside region
    img[mask == 255] = cv2.add(img[mask == 255], 50)

    return img


def generate_manipulated(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, file)

        image = cv2.imread(img_path)

        if image is None:
            continue

        tampered = simulate_tampering(image)

        cv2.imwrite(os.path.join(output_dir, file), tampered)

    print(f"Manipulated images saved to {output_dir}")