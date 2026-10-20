import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_stars(image_path, sigma_multiplier = 3.0, max_stars = 50, min_area = 3, max_area = 500):
    #1.Load the image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
   
    #2. Dynammic Adaptive Thresholding (I_thresh = M + k * o)
    mean_val, std_val = np.mean(blurred), np.std(blurred)
    threshold_val = mean_val + (sigma_multiplier * std_val)
    _, thresh_mask =cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    thresh_mask = thresh_mask.astype(np.uint8)

    #3. Find contours and spacial moments 
    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars = []
    for cnt in contours:
        M = cv2.moments(cnt)
        brightness = M["m00"]

        if brightness <= 0 or not (min_area <= brightness<= max_area):
            continue

        #Sub-pixels controid coordinates
        cx = M["m10"] / brightness
        cy = M["m01"] / brightness

        #reject ling streaks (satellites or plane )
        if len(cnt)>= 5:
            _, (d1, d2), _ = cv2.fitEllipse(cnt)
            if d1 > 0 and d2 > 0:
                aspect_ratio = min(d1, d2) / max(d1,d2)
                if aspect_ratio < 0.65:
                    continue
        stars.append((cx, cy, brightness))

    # Sort by brightness desc. and return top 
    stars.sort(key=lambda s: s[2], reverse=True)
    return stars[:max_stars],img

def visualize_results(image_path, stars):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)

    # Extract x and y lists for plotting
    x_coords = [s[0] for s in stars]
    y_coords = [s[1] for s in stars]    

    plt.scatter(x_coords, y_coords, s=60, facecolors='none', edgecolors='r', linewidth=1.5, label='Detected Stars')
    plt.title(f"Extracted Top {len(stars)} Star Centroids")
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TEST_IMAGE = "Star1_W1.jpg"

    print(f"Extracting stars from {TEST_IMAGE}...")
    detected_stars, raw_img = extract_stars(TEST_IMAGE, sigma_multiplier=3.0, max_stars=50)

    print(f"Successfully extracted {len(detected_stars)} stars. \n")
    print("Top 5 Brightest Stars (x, y, brightness):")
    for i, (x, y, b) in enumerate(detected_stars[:5], 1):
        print(f" Star{i}: x={x:.2f}, y= {y:.2f}, brightness={b:.1f}")

    visualize_results(TEST_IMAGE, detected_stars)
