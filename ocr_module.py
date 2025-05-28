import os
import cv2
import keras_ocr
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt


import cv2

def preprocess_image(frame, target_width):
    """Return the original image if target_width is None or 0; otherwise, resize it."""
    
    print("Target width: ", target_width)
    
    # Get original dimensions
    h, w = frame.shape[:2]

    # If target_width is not specified or 0, return the original image
    if target_width is None or target_width == 0:
        return frame
    
    # Prevent division by zero error
    if w == 0:
        raise ValueError("Invalid image width: 0")
    
    # Compute the new height to maintain the aspect ratio
    aspect_ratio = h / w
    new_height = int(target_width * aspect_ratio)
    
    # Resize while keeping aspect ratio
    resized_image = cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image

def init_ocr_pipeline(custom_recognizer_path=None, custom_detector_path=None, use_gpu=True):
    """
    Create and return a Keras-OCR pipeline with optional custom detector/recognizer weights.
    
    Args:
        custom_recognizer_path: Path to custom recognizer weights file
        custom_detector_path: Path to custom detector weights file
        use_gpu: Boolean flag to determine whether to use GPU or CPU
    
    Returns:
        Configured keras-OCR pipeline
    """
    import tensorflow as tf
    
    # Note: The main GPU visibility should be set via environment variable
    # before TensorFlow is imported. This function just configures memory growth.
    if use_gpu:
        print("Initialize Keras-OCR pipeline with GPU support")
        # Configure GPU memory growth to prevent OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU(s) detected: {len(gpus)}")
            except RuntimeError as e:
                print(f"Memory growth setting error: {e}")
        else:
            print("No GPUs detected despite GPU mode being enabled")
    else:
        print("Initialize Keras-OCR pipeline with CPU only")
        # GPU visibility should already be disabled via environment variable
        
    # Create the pipeline
    pipeline = keras_ocr.pipeline.Pipeline()
    
    # Load custom recognizer weights (if provided)
    if custom_recognizer_path:
        pipeline.recognizer.model.load_weights(custom_recognizer_path)

    # Load custom detector weights (if provided)
    if custom_detector_path:
        pipeline.detector.model.load_weights(custom_detector_path)
    
    return pipeline

def run_ocr_on_frame(pipeline, frame, results_folder, target_width):
    """
    Run the Keras-OCR pipeline on a single frame (image) with a custom target width.
    1. Resize the image for faster processing using the provided target_width.
    2. Recognize text and bounding boxes.
    3. Save the results to disk.
    
    Returns:
        (text_result_path, image_result_path, recognized_lines)
    """
    frame = preprocess_image(frame, target_width=target_width)  # Use the custom width
    # Recognize text (pipeline.recognize() expects a list of images)
    predictions = pipeline.recognize([frame])[0]

    # Create a filename with yy-mm-dd-hh-mm
    filename = datetime.now().strftime("%y-%m-%d-%H-%M")
    text_result_path = os.path.join(results_folder, f"{filename}.txt")
    image_result_path = os.path.join(results_folder, f"{filename}_result.png")

    # Extract recognized text and bounding boxes
    recognized_lines = []
    for (text, box) in predictions:
        box_str = ", ".join([f"({int(pt[0])}, {int(pt[1])})" for pt in box])
        recognized_lines.append(f"{text}, {box_str}")

    # Save to text file
    with open(text_result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(recognized_lines) + "\n")

    # Save annotated image
    fig, ax = plt.subplots(figsize=(10, 10))
    keras_ocr.tools.drawAnnotations(image=frame, predictions=predictions, ax=ax)
    fig.savefig(image_result_path)
    plt.close(fig)

    return text_result_path, image_result_path, recognized_lines
