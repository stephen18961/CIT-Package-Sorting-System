import os
import sys
import csv
import cv2
import numpy as np
import json
import time
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm

# Add the project directory to the system path
sys.path.append('.')

# Import the required modules from your existing codeur
from extensions import app, db
from models import Settings
from predictor import approximate_receiver_name, noise_words
from ocr_module import init_ocr_pipeline, preprocess_image

# Constants
TEST_IMAGES_DIR = "test_data"  # Directory containing test images
RESULTS_DIR = "ocr_accuracy_results"  # Directory to store results
GROUND_TRUTH_FILE = os.path.join(RESULTS_DIR, "ground_truth.json")  # File to store ground truth data

# Create directories if they don't exist
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_ground_truth():
    """Load existing ground truth data if available."""
    if os.path.exists(GROUND_TRUTH_FILE):
        try:
            with open(GROUND_TRUTH_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {GROUND_TRUTH_FILE}. Creating new ground truth file.")
    return {}

def save_ground_truth(ground_truth):
    """Save ground truth data to file."""
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=4)

def get_candidates_from_db():
    """Get candidate names from the database."""
    with app.app_context():
        from models import Student, Staff
        students = [(s.name, str(s.floor)) for s in Student.query.all()]
        staff = [(s.name, str(s.floor)) for s in Staff.query.all()]
        return students + staff

def process_image_with_model(image_path, pipeline, settings, candidates):
    """
    Process a single image with the given OCR pipeline and settings.
    
    Args:
        image_path: Path to the image file
        pipeline: Initialized OCR pipeline
        settings: Settings object with processing parameters
        candidates: List of candidate names for matching
        
    Returns:
        dict: Results of the OCR and name matching process including timing information
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": f"Failed to read image: {image_path}"}
    
    # Preprocess the image - measure preprocessing time
    preprocess_start = time.time()
    preprocessed_frame = preprocess_image(frame, target_width=settings.ocr_preprocess_width)
    preprocess_time = time.time() - preprocess_start
    
    # Run OCR with the pipeline - measure OCR time
    try:
        # Recognize text (pipeline.recognize() expects a list of images)
        ocr_start = time.time()
        predictions = pipeline.recognize([preprocessed_frame])[0]
        ocr_time = time.time() - ocr_start
        
        # Extract recognized text
        recognized_lines = []
        for (text, box) in predictions:
            box_str = ", ".join([f"({int(pt[0])}, {int(pt[1])})" for pt in box])
            recognized_lines.append(f"{text}, {box_str}")
        
        ocr_text_combined = "\n".join(recognized_lines)
        
        # Run the rule-based predictor on the OCR result - measure matching time
        matching_start = time.time()
        best_match, score = approximate_receiver_name(
            ocr_text_combined,
            candidates,
            noise_words,
            threshold=0.8,  # Noise word threshold
            match_threshold=settings.match_threshold  # Match threshold from settings
        )
        matching_time = time.time() - matching_start
        
        # Calculate total processing time
        total_time = preprocess_time + ocr_time + matching_time
        
        if best_match:
            name_detected = best_match[0]
            floor_detected = best_match[1]
            return {
                "status": "Success",
                "receiver_name": name_detected,
                "target_floor": floor_detected,
                "confidence_score": score,
                "ocr_text": ocr_text_combined,
                "threshold_used": settings.match_threshold,
                "timing": {
                    "preprocess_time": preprocess_time,
                    "ocr_inference_time": ocr_time,
                    "name_matching_time": matching_time,
                    "total_processing_time": total_time
                }
            }
        else:
            return {
                "status": "Failed",
                "error": "No matching name found",
                "ocr_text": ocr_text_combined,
                "confidence_score": score,
                "threshold_used": settings.match_threshold,
                "timing": {
                    "preprocess_time": preprocess_time,
                    "ocr_inference_time": ocr_time,
                    "name_matching_time": matching_time,
                    "total_processing_time": total_time
                }
            }
            
    except Exception as e:
        return {"status": "Error", "error": str(e)}

def run_accuracy_test():
    """
    Run accuracy tests on the trained OCR model with timing information.
    """
    print("\n--- Starting OCR Accuracy Testing ---")
    
    # Load ground truth data
    ground_truth = load_ground_truth()
    
    # Get settings from the database
    with app.app_context():
        settings = Settings.query.first()
        if not settings:
            settings = Settings(
                ocr_preprocess_width=0,
                match_threshold=0.36,
                use_gpu=False
            )
            db.session.add(settings)
            db.session.commit()
    
    # Check for GPU availability
    gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    print(f"GPU Available: {gpu_available}")
    
    # Get candidate names from the database
    candidates = get_candidates_from_db()
    print(f"Loaded {len(candidates)} candidates from database")
    
    # Initialize only the trained OCR pipeline
    print("Initializing trained OCR model...")
    model_init_start = time.time()
    trained_pipeline = init_ocr_pipeline(
        custom_recognizer_path='models/recognizer_with_augment.h5',
        custom_detector_path=None,
        use_gpu=settings.use_gpu and gpu_available
    )
    model_init_time = time.time() - model_init_start
    print(f"Model initialization time: {model_init_time:.2f} seconds")
    
    # Get list of test images
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {TEST_IMAGES_DIR}. Please add test images.")
        return
    
    print(f"Found {len(image_files)} test images.")
    
    # Prepare results data
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "ocr_preprocess_width": settings.ocr_preprocess_width,
            "match_threshold": settings.match_threshold,
            "use_gpu": settings.use_gpu and gpu_available
        },
        "model_initialization_time": model_init_time,
        "images": {}
    }
    
    # Track timing metrics
    total_ocr_time = 0
    total_preprocess_time = 0
    total_matching_time = 0
    total_overall_time = 0
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        image_id = os.path.splitext(image_file)[0]
        
        # Check if ground truth exists for this image
        gt_exists = image_id in ground_truth
        if gt_exists:
            gt_name = ground_truth[image_id]["name"]
            gt_floor = ground_truth[image_id]["floor"]
            print(f"\nImage {image_id} has ground truth: {gt_name} (Floor: {gt_floor})")
        else:
            print(f"\nNo ground truth for image {image_id}. Will need manual verification.")
        
        # Process with trained model
        trained_result = process_image_with_model(
            image_path, trained_pipeline, settings, candidates
        )
        
        # Update timing metrics
        if "timing" in trained_result:
            total_ocr_time += trained_result["timing"]["ocr_inference_time"]
            total_preprocess_time += trained_result["timing"]["preprocess_time"]
            total_matching_time += trained_result["timing"]["name_matching_time"]
            total_overall_time += trained_result["timing"]["total_processing_time"]
        
        # Store results
        results["images"][image_id] = {
            "file_name": image_file,
            "ground_truth_exists": gt_exists,
            "ground_truth": ground_truth.get(image_id, {}),
            "trained_model_result": trained_result
        }
        
        # Display results for verification
        print(f"\nResults for {image_file}:")
        print(f"  Trained Model: {trained_result.get('receiver_name', 'None')} (Score: {trained_result.get('confidence_score', 0):.4f})")
        if "timing" in trained_result:
            print(f"  OCR inference time: {trained_result['timing']['ocr_inference_time']:.4f} seconds")
            print(f"  Total processing time: {trained_result['timing']['total_processing_time']:.4f} seconds")
        
        # If no ground truth exists, prompt for verification
        if not gt_exists:
            print("\nPlease verify the prediction:")
            print("1. Trained Model is correct")
            print("2. Trained Model is incorrect (Enter correct name)")
            print("3. Skip this image")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == "1":
                # Use trained model prediction as ground truth
                ground_truth[image_id] = {
                    "name": trained_result.get("receiver_name"),
                    "floor": trained_result.get("target_floor"),
                    "source": "trained_model"
                }
                print(f"Set ground truth to trained model prediction: {ground_truth[image_id]['name']}")
                
            elif choice == "2":
                # Enter correct name manually
                correct_name = input("Enter correct name: ")
                correct_floor = input("Enter correct floor: ")
                ground_truth[image_id] = {
                    "name": correct_name,
                    "floor": correct_floor,
                    "source": "manual"
                }
                print(f"Set ground truth to manual entry: {ground_truth[image_id]['name']}")
                
            elif choice == "3":
                print("Skipping this image")
            
            else:
                print("Invalid choice. Skipping this image.")
    
    # Calculate accuracy metrics
    trained_correct = 0
    total_with_gt = 0
    
    for image_id, image_data in results["images"].items():
        if image_id in ground_truth:
            total_with_gt += 1
            gt_name = ground_truth[image_id]["name"]
            
            trained_result = image_data["trained_model_result"]
            
            if trained_result.get("status") == "Success" and trained_result.get("receiver_name") == gt_name:
                trained_correct += 1
    
    # Calculate accuracy percentage
    if total_with_gt > 0:
        trained_accuracy = (trained_correct / total_with_gt) * 100
    else:
        trained_accuracy = 0
    
    # Calculate average timing metrics
    num_images = len(image_files)
    avg_ocr_time = total_ocr_time / num_images if num_images > 0 else 0
    avg_preprocess_time = total_preprocess_time / num_images if num_images > 0 else 0
    avg_matching_time = total_matching_time / num_images if num_images > 0 else 0
    avg_overall_time = total_overall_time / num_images if num_images > 0 else 0
    
    # Add metrics to results
    results["metrics"] = {
        "total_images": len(image_files),
        "images_with_ground_truth": total_with_gt,
        "trained_model_correct": trained_correct,
        "trained_model_accuracy": trained_accuracy,
        "timing": {
            "model_initialization_time": model_init_time,
            "average_ocr_inference_time": avg_ocr_time,
            "average_preprocess_time": avg_preprocess_time,
            "average_name_matching_time": avg_matching_time,
            "average_total_processing_time": avg_overall_time,
            "total_ocr_inference_time": total_ocr_time,
            "total_processing_time": total_overall_time
        }
    }
    
    # Save ground truth data
    save_ground_truth(ground_truth)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"ocr_accuracy_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate CSV summary
    csv_file = os.path.join(RESULTS_DIR, f"ocr_accuracy_summary_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image ID', 'Ground Truth', 'Trained Model Prediction', 'Confidence Score', 'Correct', 'OCR Time (s)', 'Total Time (s)'])
        
        for image_id, image_data in results["images"].items():
            if image_id in ground_truth:
                gt_name = ground_truth[image_id]["name"]
                trained_result = image_data["trained_model_result"]
                
                trained_name = trained_result.get("receiver_name", "N/A")
                trained_score = trained_result.get("confidence_score", 0)
                
                trained_correct = 'Yes' if trained_name == gt_name else 'No'
                
                ocr_time = trained_result.get("timing", {}).get("ocr_inference_time", 0)
                total_time = trained_result.get("timing", {}).get("total_processing_time", 0)
                
                writer.writerow([image_id, gt_name, trained_name, f"{trained_score:.4f}", trained_correct, f"{ocr_time:.4f}", f"{total_time:.4f}"])
    
    # Generate timing CSV
    timing_csv_file = os.path.join(RESULTS_DIR, f"ocr_timing_details_{timestamp}.csv")
    with open(timing_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Image ID', 
            'Preprocess Time (s)', 
            'OCR Inference Time (s)', 
            'Name Matching Time (s)', 
            'Total Processing Time (s)'
        ])
        
        for image_id, image_data in results["images"].items():
            trained_result = image_data["trained_model_result"]
            if "timing" in trained_result:
                writer.writerow([
                    image_id,
                    f"{trained_result['timing']['preprocess_time']:.4f}",
                    f"{trained_result['timing']['ocr_inference_time']:.4f}",
                    f"{trained_result['timing']['name_matching_time']:.4f}",
                    f"{trained_result['timing']['total_processing_time']:.4f}"
                ])
    
    # Print summary
    print("\n--- OCR Accuracy Test Summary ---")
    print(f"Total images tested: {len(image_files)}")
    print(f"Images with ground truth: {total_with_gt}")
    print(f"Trained Model accuracy: {trained_accuracy:.2f}% ({trained_correct}/{total_with_gt})")
    print("\n--- Timing Information ---")
    print(f"Model initialization time: {model_init_time:.4f} seconds")
    print(f"Average OCR inference time: {avg_ocr_time:.4f} seconds")
    print(f"Average preprocessing time: {avg_preprocess_time:.4f} seconds")
    print(f"Average name matching time: {avg_matching_time:.4f} seconds")
    print(f"Average total processing time: {avg_overall_time:.4f} seconds")
    print(f"\nResults saved to: {results_file}")
    print(f"CSV summary saved to: {csv_file}")
    print(f"Timing details saved to: {timing_csv_file}")
    print("--- Testing Complete ---")

if __name__ == "__main__":
    # Check if test images directory has images
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Creating test images directory: {TEST_IMAGES_DIR}")
        print("Please add test images to this directory and run the script again.")
        os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
        sys.exit(0)
        
    image_count = len([f for f in os.listdir(TEST_IMAGES_DIR) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if image_count == 0:
        print(f"No images found in {TEST_IMAGES_DIR}")
        print("Please add test images to this directory and run the script again.")
        sys.exit(0)
    
    # Run the accuracy test
    run_accuracy_test()