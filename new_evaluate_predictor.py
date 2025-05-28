import os
import sys
import csv
import json
from datetime import datetime
from tqdm import tqdm

# Add the project directory to the system path
sys.path.append('.')

# Import the required modules from your existing code
from extensions import app, db
from models import Settings
from predictor import approximate_receiver_name, noise_words

# Constants
RESULTS_DIR = "ocr_accuracy_results"  # Directory containing previous OCR results
PREDICTOR_RESULTS_DIR = "predictor_evaluation_results"  # Directory to store predictor evaluation results
GROUND_TRUTH_FILE = os.path.join(RESULTS_DIR, "ground_truth.json")  # File with ground truth data
THRESHOLD = 0.36  # Fixed threshold for evaluation

# Create results directory if it doesn't exist
os.makedirs(PREDICTOR_RESULTS_DIR, exist_ok=True)

def get_latest_ocr_results():
    """Get the most recent OCR results file from the results directory."""
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("ocr_accuracy_results_") and f.endswith(".json")]
    if not result_files:
        print(f"No OCR result files found in {RESULTS_DIR}. Please run the OCR accuracy test first.")
        return None
    
    # Sort by timestamp (newest first)
    result_files.sort(reverse=True)
    latest_file = os.path.join(RESULTS_DIR, result_files[0])
    print(f"Using latest OCR results file: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading {latest_file}. Invalid JSON format.")
        return None

def load_ground_truth():
    """Load existing ground truth data."""
    if os.path.exists(GROUND_TRUTH_FILE):
        try:
            with open(GROUND_TRUTH_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {GROUND_TRUTH_FILE}.")
            return {}
    else:
        print(f"Ground truth file not found: {GROUND_TRUTH_FILE}")
        return {}

def get_candidates_from_db():
    """Get candidate names from the database."""
    with app.app_context():
        from models import Student, Staff
        students = [(s.name, str(s.floor)) for s in Student.query.all()]
        staff = [(s.name, str(s.floor)) for s in Staff.query.all()]
        return students + staff

def evaluate_predictor(ocr_results, ground_truth, candidates):
    """
    Evaluate the predictor performance using a fixed threshold.
    
    Args:
        ocr_results: Dictionary containing OCR results
        ground_truth: Dictionary with ground truth data
        candidates: List of candidate names for matching
        
    Returns:
        dict: Results of the predictor evaluation
    """
    evaluation_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": THRESHOLD,
        "overall_stats": {
            "trained_ocr": {
                "correct": 0,
                "incorrect": 0,
                "no_match": 0,
                "total": 0,
                "accuracy": 0.0
            }
        },
        "images": {}
    }
    
    # Process each image
    images_with_gt = 0
    for image_id, image_data in tqdm(ocr_results["images"].items(), desc="Evaluating images"):
        if image_id not in ground_truth:
            continue
        
        images_with_gt += 1
        gt_name = ground_truth[image_id]["name"]
        gt_floor = ground_truth[image_id]["floor"]
        
        # Get OCR text from trained model only
        trained_ocr_text = image_data["trained_model_result"].get("ocr_text", "")
        
        # Skip images with empty OCR text
        if not trained_ocr_text:
            continue
        
        # Run predictor on trained OCR text
        trained_match, trained_score = approximate_receiver_name(
            trained_ocr_text,
            candidates,
            noise_words,
            threshold=0.8,  # Noise word threshold (fixed)
            match_threshold=THRESHOLD
        )
        
        # Store results for this image
        image_results = {
            "ground_truth": {
                "name": gt_name,
                "floor": gt_floor
            },
            "trained_ocr": {
                "text": trained_ocr_text,
                "match_found": trained_match is not None,
                "predicted_name": trained_match[0] if trained_match else None,
                "predicted_floor": trained_match[1] if trained_match else None,
                "confidence_score": trained_score,
                "is_correct": trained_match is not None and trained_match[0] == gt_name
            }
        }
        
        evaluation_results["images"][image_id] = image_results
        
        # Update trained OCR statistics
        evaluation_results["overall_stats"]["trained_ocr"]["total"] += 1
        if not trained_match:
            evaluation_results["overall_stats"]["trained_ocr"]["no_match"] += 1
        elif trained_match[0] == gt_name:
            evaluation_results["overall_stats"]["trained_ocr"]["correct"] += 1
        else:
            evaluation_results["overall_stats"]["trained_ocr"]["incorrect"] += 1
    
    # Calculate accuracy for trained model
    trained_stats = evaluation_results["overall_stats"]["trained_ocr"]
    if trained_stats["total"] > 0:
        trained_stats["accuracy"] = (trained_stats["correct"] / trained_stats["total"]) * 100
    
    # Add summary
    evaluation_results["summary"] = {
        "total_images": len(ocr_results["images"]),
        "images_with_ground_truth": images_with_gt,
        "images_evaluated": trained_stats["total"],
        "trained_ocr_accuracy": trained_stats["accuracy"]
    }
    
    return evaluation_results

def analyze_errors(evaluation_results):
    """
    Analyze error patterns in the predictor results.
    
    Args:
        evaluation_results: Dictionary with predictor evaluation results
        
    Returns:
        dict: Error analysis results
    """
    error_analysis = {
        "trained_ocr": {
            "failure_cases": [],
            "common_errors": {}
        }
    }
    
    # Analyze each image result
    for image_id, image_data in evaluation_results["images"].items():
        gt_name = image_data["ground_truth"]["name"]
        
        # Analyze trained OCR errors
        trained_result = image_data["trained_ocr"]
        if not trained_result["is_correct"]:
            error_case = {
                "image_id": image_id,
                "ground_truth": gt_name,
                "predicted": trained_result["predicted_name"],
                "ocr_text": trained_result["text"],
                "confidence": trained_result["confidence_score"]
            }
            error_analysis["trained_ocr"]["failure_cases"].append(error_case)
            
            # Track error pattern
            if trained_result["predicted_name"]:
                error_pair = f"{gt_name} -> {trained_result['predicted_name']}"
                if error_pair not in error_analysis["trained_ocr"]["common_errors"]:
                    error_analysis["trained_ocr"]["common_errors"][error_pair] = {
                        "count": 0,
                        "images": []
                    }
                
                error_analysis["trained_ocr"]["common_errors"][error_pair]["count"] += 1
                error_analysis["trained_ocr"]["common_errors"][error_pair]["images"].append(image_id)
    
    # Sort common errors by frequency
    trained_sorted_errors = sorted(
        error_analysis["trained_ocr"]["common_errors"].items(), 
        key=lambda item: item[1]["count"], 
        reverse=True
    )[:10]
    
    # Replace with sorted dictionary
    error_analysis["trained_ocr"]["common_errors"] = dict(trained_sorted_errors)
    
    return error_analysis

def run_predictor_evaluation():
    """
    Run the predictor evaluation using previously generated OCR results.
    """
    print("\n--- Starting Predictor Evaluation (Trained Model Only, Threshold: {}) ---".format(THRESHOLD))
    
    # Load the latest OCR results
    ocr_results = get_latest_ocr_results()
    if not ocr_results:
        return
    
    # Load ground truth data
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("No ground truth data found. Unable to evaluate predictor.")
        return
    
    # Get candidate names from the database
    candidates = get_candidates_from_db()
    print(f"Loaded {len(candidates)} candidates from database")
    
    # Evaluate predictor with fixed threshold
    evaluation_results = evaluate_predictor(ocr_results, ground_truth, candidates)
    
    # Analyze errors
    error_analysis = analyze_errors(evaluation_results)
    evaluation_results["error_analysis"] = error_analysis
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(PREDICTOR_RESULTS_DIR, f"predictor_evaluation_trained_only_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    # Generate CSV summary
    csv_file = os.path.join(PREDICTOR_RESULTS_DIR, f"predictor_evaluation_summary_trained_only_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image ID', 'Ground Truth', 'Trained OCR Prediction', 'Trained Score', 'Trained Correct'])
        
        for image_id, image_data in evaluation_results["images"].items():
            gt_name = image_data["ground_truth"]["name"]
            
            trained_prediction = image_data["trained_ocr"]["predicted_name"] or "No match"
            trained_score = image_data["trained_ocr"]["confidence_score"]
            trained_correct = "Yes" if image_data["trained_ocr"]["is_correct"] else "No"
            
            writer.writerow([
                image_id, gt_name, 
                trained_prediction, f"{trained_score:.4f}", trained_correct
            ])
    
    # Generate detailed error CSV
    error_csv_file = os.path.join(PREDICTOR_RESULTS_DIR, f"predictor_errors_trained_only_{timestamp}.csv")
    with open(error_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image ID', 'Ground Truth', 'Prediction', 'Confidence', 'OCR Text'])
        
        # Add trained OCR errors
        for error in error_analysis["trained_ocr"]["failure_cases"]:
            writer.writerow([
                error["image_id"],
                error["ground_truth"], error["predicted"] or "No match", 
                f"{error['confidence']:.4f}", error["ocr_text"]#[:50]  # Truncate text for readability
            ])
    
    # Generate a CSV file specifically for common errors with image IDs
    common_errors_csv_file = os.path.join(PREDICTOR_RESULTS_DIR, f"common_errors_trained_only_{timestamp}.csv")
    with open(common_errors_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Error Pattern', 'Count', 'Image IDs'])
        
        # Add trained OCR common errors
        for error_pair, error_data in error_analysis["trained_ocr"]["common_errors"].items():
            writer.writerow([
                error_pair, 
                error_data["count"],
                ', '.join(error_data["images"])
            ])
    
    # Print summary
    trained_stats = evaluation_results["overall_stats"]["trained_ocr"]
    
    print("\n--- Predictor Evaluation Summary (Trained Model Only) ---")
    print(f"Threshold used: {THRESHOLD}")
    print(f"Total images with ground truth: {evaluation_results['summary']['images_with_ground_truth']}")
    print(f"Images evaluated: {trained_stats['total']}")
    
    print("\nTrained OCR Results:")
    print(f"  Correct predictions: {trained_stats['correct']} ({trained_stats['correct']/trained_stats['total']*100:.2f}%)")
    print(f"  Incorrect predictions: {trained_stats['incorrect']} ({trained_stats['incorrect']/trained_stats['total']*100:.2f}%)")
    print(f"  No match found: {trained_stats['no_match']} ({trained_stats['no_match']/trained_stats['total']*100:.2f}%)")
    print(f"  Overall accuracy: {trained_stats['accuracy']:.2f}%")
    
    # Print common errors with image IDs
    print("\nMost Common Errors (Trained OCR):")
    for error_pair, error_data in list(error_analysis["trained_ocr"]["common_errors"].items())[:5]:
        print(f"  {error_pair}: {error_data['count']} occurrences")
        print(f"    Image IDs: {', '.join(error_data['images'])}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Summary CSV saved to: {csv_file}")
    print(f"Error analysis saved to: {error_csv_file}")
    print(f"Common errors with image IDs saved to: {common_errors_csv_file}")
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    # Check if results directory exists and has OCR results
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Please run the OCR accuracy test first to generate OCR results.")
        sys.exit(1)
    
    ocr_result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("ocr_accuracy_results_") and f.endswith(".json")]
    if not ocr_result_files:
        print(f"No OCR result files found in {RESULTS_DIR}.")
        print("Please run the OCR accuracy test first to generate OCR results.")
        sys.exit(1)
    
    # Run the predictor evaluation
    run_predictor_evaluation()