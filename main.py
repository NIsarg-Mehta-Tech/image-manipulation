import cv2 as cv
import numpy as np
import os

save_dir = "Capture_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_counter = 0

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame,1)
    if not ret:
        print("Failed to get the frame")
        break

    cv.imshow("Frame", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key%256 == 32:
        img_name = "frame_{}.jpg".format(img_counter)
        cv.imwrite(os.path.join(save_dir, img_name), frame)
        cv.imshow("Original File", frame)
        print("Frame saved as", img_name)
        img_counter += 1

cap.release()
cv.destroyAllWindows()

image_list = []
image_files = sorted([f for f in os.listdir(save_dir) 
                      if f.endswith((".jpg", ".jpeg", ".png"))
                    ])

for file in image_files:
    image_path = os.path.join(save_dir, file)
    img = cv.imread(image_path)

    if img is not None:
        image_list.append(img)
        print(f"Loaded: {file}")
    else:
        print(f"Error opening image {file}")

processed_images = []

rows, cols = image_list[0].shape[:2]

titles = ["Gray", "Rotated", "Cropped", "Translated", "Gaussian Blur"]

for idx, img in enumerate(image_list):
    
    if idx == 0:
        processed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        processed = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
        cv.imshow(titles[idx], processed)
        label = titles[idx]
    
    elif idx == 1:

        center = (cols // 2, rows // 2)
        angle = -90
        scale = 1.0

        M = cv.getRotationMatrix2D(center, angle, scale)
        processed = cv.warpAffine(img, M, (cols, rows))
        cv.imshow(titles[idx], processed)
        label = titles[idx]

    elif idx == 2:
        processed = img[100:(cols-290), 200:(rows-500)]
        cv.imshow(titles[idx], processed)
        label = titles[idx]

    elif idx == 3:
        T = np.float32([[1, 0, 100], 
                [0, 1, 50]])

        processed =  cv.warpAffine(img, T, (cols, rows))
        cv.imshow(titles[idx], processed)

        label = titles[idx]
    
    elif idx == 4:
        processed = cv.GaussianBlur(img, (7,7),10)
        cv.imshow(titles[idx], processed)
        label = titles[idx]

    processed = cv.resize(processed, (cols, rows))
    cv.putText(processed, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
               1, (0, 255, 0), 2, cv.LINE_AA)

    processed_images.append(processed)

combined_image = np.hstack(processed_images)
cv.imshow("Combined Image", combined_image)

cv.waitKey(0)
cv.destroyAllWindows()