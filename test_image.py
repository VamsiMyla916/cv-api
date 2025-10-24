from ultralytics import YOLO
import cv2

print("Loading model 'yolov8n.pt'...")
# Load the pre-trained YOLOv8 'nano' model
model = YOLO('yolov8n.pt') 

# Define the image you want to test
image_path = 'my_test_image.jpg' # <-- Make sure this is the correct file name
print(f"Processing image: {image_path}")

# Run inference on the image
# We'll set a low confidence (conf=0.2) to catch uncertain detections
results = model(image_path, conf=0.2) 

# --- NEW DEBUGGING PART ---
print("\n--- Detection Results ---")
# Get the first (and only) result object
result = results[0]

# Check if any boxes were detected
if len(result.boxes) == 0:
    print("No objects detected.")
else:
    print(f"Detected {len(result.boxes)} objects.")
    # Loop over each detected box
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"-> Found: {class_name}, Confidence: {confidence:.2f}")
print("---------------------------\n")
# --- END OF DEBUGGING PART ---

# Get the annotated image (with boxes drawn on it)
annotated_image = result.plot()

# Save the annotated image to a file
output_save_path = 'annotated_output.jpg'
cv2.imwrite(output_save_path, annotated_image)
print(f"SUCCESS: Saved annotated image to: {output_save_path}")

# Display the annotated image in a window
cv2.imshow("YOLOv8 Detection", annotated_image)
print("Showing image in a new window. Press any key to close...")

# Wait for a key press and then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done.")