import cv2
import numpy as np
import os
import random


# ===================== BASIC MANIPULATIONS =====================

def subtle_brightness(image, mask):
    img = image.copy().astype(np.float32)
    factor = random.uniform(1.05, 1.20)
    img[mask == 255] *= factor
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def gaussian_blur_region(image, mask):
    ksize = random.choice([9, 15, 21])
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    img = image.copy()
    img[mask == 255] = blurred[mask == 255]
    return img


def add_noise_region(image, mask):
    img = image.copy().astype(np.float32)
    sigma = random.uniform(5, 20)
    noise = np.random.normal(0, sigma, img.shape)
    img[mask == 255] += noise[mask == 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ===================== NEW MANIPULATIONS =====================

def contrast_adjust_region(image, mask):
    """Locally adjust contrast in the masked region."""
    img = image.copy().astype(np.float32)
    alpha = random.uniform(0.7, 1.4)  # contrast factor
    mean_val = img[mask == 255].mean()
    img[mask == 255] = alpha * (img[mask == 255] - mean_val) + mean_val
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def copy_move_region(image, mask, center):
    """Copy a region from elsewhere in the image and paste it."""
    img = image.copy()
    h, w = img.shape[:2]
    # Pick a source offset
    dx = random.randint(-w // 4, w // 4)
    dy = random.randint(-h // 4, h // 4)
    ys, xs = np.where(mask == 255)
    src_ys = np.clip(ys + dy, 0, h - 1)
    src_xs = np.clip(xs + dx, 0, w - 1)
    img[ys, xs] = image[src_ys, src_xs]
    return img


def inpainting_region(image, mask):
    """Use OpenCV inpainting to remove content in the masked region."""
    return cv2.inpaint(image, mask, inpaintRadius=5,
                       flags=cv2.INPAINT_TELEA)


def jpeg_artifact_region(image, mask):
    """Introduce JPEG compression artifacts in a region."""
    img = image.copy()
    quality = random.randint(10, 40)
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img[mask == 255] = compressed[mask == 255]
    return img


def elastic_warp_region(image, mask):
    """Apply slight elastic deformation to the masked region."""
    img = image.copy()
    h, w = image.shape[:2]
    alpha = random.uniform(8, 15)
    sigma = random.uniform(3, 5)
    dx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32),
                          (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    img[mask == 255] = warped[mask == 255]
    return img


# ===================== MASK GENERATION =====================

def make_mask(h, w, shape="circle"):
    """Generate a random mask with varied shapes and sizes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = random.randint(w // 5, 4 * w // 5)
    cy = random.randint(h // 5, 4 * h // 5)

    if shape == "circle":
        radius = random.randint(25, 60)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    elif shape == "ellipse":
        axes = (random.randint(20, 60), random.randint(15, 45))
        angle = random.randint(0, 180)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
    elif shape == "irregular":
        # Random polygon
        n_pts = random.randint(5, 8)
        pts = []
        for _ in range(n_pts):
            px = cx + random.randint(-40, 40)
            py = cy + random.randint(-40, 40)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Soft edges via Gaussian blur
    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask, (cx, cy)


# ===================== MAIN TAMPERING =====================

MANIPULATIONS = [
    "brightness", "blur", "noise",
    "contrast", "copy_move", "inpaint",
    "jpeg_artifact", "elastic_warp"
]


def simulate_realistic_tampering(image):
    img = image.copy()
    h, w = img.shape[:2]

    shape = random.choice(["circle", "ellipse", "irregular"])
    mask, center = make_mask(h, w, shape)

    # Apply 1–2 manipulations for compound tampering
    n_ops = random.choices([1, 2], weights=[0.6, 0.4])[0]
    choices = random.sample(MANIPULATIONS, n_ops)

    for choice in choices:
        if choice == "brightness":
            img = subtle_brightness(img, mask)
        elif choice == "blur":
            img = gaussian_blur_region(img, mask)
        elif choice == "noise":
            img = add_noise_region(img, mask)
        elif choice == "contrast":
            img = contrast_adjust_region(img, mask)
        elif choice == "copy_move":
            img = copy_move_region(img, mask, center)
        elif choice == "inpaint":
            img = inpainting_region(img, mask)
        elif choice == "jpeg_artifact":
            img = jpeg_artifact_region(img, mask)
        elif choice == "elastic_warp":
            img = elastic_warp_region(img, mask)

    return img


def generate_manipulated(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        tampered = simulate_realistic_tampering(image)
        cv2.imwrite(os.path.join(output_dir, file), tampered)

    print(f"Manipulated images saved to {output_dir}")