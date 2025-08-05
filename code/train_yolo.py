from ultralytics import YOLO

# yolo github page: https://github.com/ultralytics/ultralytics

# Load a pretrained YOLO11n model
model = YOLO("yolo11x.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="../materials/2025-08-04/extract_frames/first_try/data.yaml",  # Path to dataset configuration file
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="mps",
    project="./yolo_training",          # Save in current directory
    name="fitting" # Specific folder name
)

# Evaluate the model's performance on the validation set
# metrics = model.val()

# Perform object detection on an image
# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results

# Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model
# print(path)