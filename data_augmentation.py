import cv2
import os
import argparse

def augment_data(input_dir, output_dir):
    """Augment images with rotations and flips."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(input_dir, filename))
            # Rotate
            cv2.imwrite(os.path.join(output_dir, f"rot_{filename}"), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
            # Flip
            cv2.imwrite(os.path.join(output_dir, f"flip_{filename}"), cv2.flip(img, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lib/data")
    parser.add_argument("--output", default="lib/data_augmented")
    args = parser.parse_args()
    augment_data(args.data, args.output)