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
from predictor_detailed import approximate_receiver_name, noise_words

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

# Modified function to get detailed scores from approximate_receiver_name
def get_detailed_prediction(ocr_text, candidates, noise_words, threshold=0.8, match_threshold=0.36):
    """
    Get detailed prediction scores from the approximate_receiver_name function.
    This is a wrapper that will call the original function but return more detailed scoring info.
    """
    # First, call the modified version of approximate_receiver_name that returns detailed scores
    # You'll need to modify your original function to return these detailed scores
    match, final_score, detailed_scores = approximate_receiver_name(
        ocr_text,
        candidates,
        noise_words,
        threshold=threshold,
        match_threshold=match_threshold
    )
    
    return match, final_score, detailed_scores

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
        
        # Run predictor on trained OCR text with detailed scores
        trained_match, trained_score, detailed_scores = approximate_receiver_name(
            trained_ocr_text,
            candidates,
            noise_words,
            threshold=0.8,  # Noise word threshold (fixed)
            match_threshold=THRESHOLD,
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
                "detailed_scores": detailed_scores,  # Include all detailed scores
                "is_correct": trained_match is not None and trained_match[0] == gt_name,
                "candidate_scores": detailed_scores.get("candidate_scores", [])  # Include all candidate scores
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
                "confidence": trained_result["confidence_score"],
                "detailed_scores": trained_result["detailed_scores"]  # Include detailed scores
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

def generate_detailed_csv(evaluation_results, timestamp):
    """
    Generate detailed CSV files with all score components in a tidier format.
    
    Args:
        evaluation_results: Dictionary with predictor evaluation results
        timestamp: Timestamp string for file naming
        
    Returns:
        List of file paths that were created
    """
    csv_files = []
    
    # 1. Generate summary CSV with all score components
    csv_file = os.path.join(PREDICTOR_RESULTS_DIR, f"predictor_evaluation_summary_trained_only_{timestamp}.csv")
    csv_files.append(csv_file)
    
    # Define columns we want to exclude from the CSV (text lists that clutter the output)
    exclude_columns = ['original_words', 'cleaned_words', 'enhanced_words', 'candidate_scores']
    
    # Find all possible detailed score keys across all images (filtering out excluded ones)
    detailed_score_keys = set()
    for image_id, image_data in evaluation_results["images"].items():
        ocr_data = image_data["trained_ocr"]
        if "detailed_scores" in ocr_data:
            # Add top-level keys only, excluding text lists
            basic_keys = [k for k in ocr_data["detailed_scores"].keys() 
                         if k not in exclude_columns and not isinstance(ocr_data["detailed_scores"][k], dict)]
            detailed_score_keys.update(basic_keys)
    
    # Sort keys for consistent column order
    detailed_score_keys = sorted(detailed_score_keys)
    
    with open(csv_file, 'w', newline='') as f:
        # Create header row with all detailed score columns (without "Score:" prefix)
        header = ['Image ID', 'Ground Truth', 'Trained OCR Prediction', 'Final Score', 'Trained Correct']
        for key in detailed_score_keys:
            header.append(key)
        
        # Add columns for top matches (up to 3)
        for i in range(1, 4):
            header.extend([f'top_match_name_{i}', f'top_match_floor_{i}', f'top_match_score_{i}'])
        
        writer = csv.writer(f)
        writer.writerow(header)
        
        for image_id, image_data in evaluation_results["images"].items():
            gt_name = image_data["ground_truth"]["name"]
            
            trained_prediction = image_data["trained_ocr"]["predicted_name"] or "No match"
            trained_score = image_data["trained_ocr"]["confidence_score"]
            trained_correct = "Yes" if image_data["trained_ocr"]["is_correct"] else "No"
            
            # Start with basic columns
            row = [
                image_id, gt_name, 
                trained_prediction, f"{trained_score:.4f}", trained_correct
            ]
            
            # Add detailed scores (numerical and boolean values only)
            detailed_scores = image_data["trained_ocr"].get("detailed_scores", {})
            for key in detailed_score_keys:
                if key in detailed_scores:
                    value = detailed_scores[key]
                    if isinstance(value, (float, int, bool)):
                        # Format floating point values to 4 decimal places
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    elif isinstance(value, list) and key == "potential_names" and value:
                        # For potential names, just show the first one instead of a list
                        row.append(value[0] if value else "")
                    else:
                        row.append("")
                else:
                    row.append("")
            
            # Add top matches (up to 3) in separate columns
            top_matches = detailed_scores.get("top_matches", [])
            for i in range(3):  # Add up to 3 top matches
                if i < len(top_matches):
                    match = top_matches[i]
                    row.extend([match.get("name", ""), match.get("floor", ""), f"{match.get('score', 0):.4f}"])
                else:
                    row.extend(["", "", ""])
            
            writer.writerow(row)
    
    # 2. Generate detailed error CSV with all scores (tidy version)
    error_csv_file = os.path.join(PREDICTOR_RESULTS_DIR, f"predictor_errors_trained_only_{timestamp}.csv")
    csv_files.append(error_csv_file)
    
    with open(error_csv_file, 'w', newline='') as f:
        # Create header row (without "Score:" prefix)
        header = ['Image ID', 'Ground Truth', 'Prediction', 'Final Confidence', 'OCR Text Length']
        for key in detailed_score_keys:
            header.append(key)
        
        # Add top matches columns
        for i in range(1, 4):
            header.extend([f'top_match_name_{i}', f'top_match_floor_{i}', f'top_match_score_{i}'])
        
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Get error cases
        error_cases = []
        for image_id, image_data in evaluation_results["images"].items():
            if not image_data["trained_ocr"]["is_correct"]:
                error_cases.append((image_id, image_data))
        
        # Add trained OCR errors with detailed scores
        for image_id, image_data in error_cases:
            ocr_data = image_data["trained_ocr"]
            
            # Start with basic columns (replace full OCR text with just the length)
            row = [
                image_id,
                image_data["ground_truth"]["name"], 
                ocr_data["predicted_name"] or "No match", 
                f"{ocr_data['confidence_score']:.4f}", 
                len(ocr_data["text"])  # Just show text length instead of full text
            ]
            
            # Add detailed scores (same approach as summary CSV)
            detailed_scores = ocr_data.get("detailed_scores", {})
            for key in detailed_score_keys:
                if key in detailed_scores:
                    value = detailed_scores[key]
                    if isinstance(value, (float, int, bool)):
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    elif isinstance(value, list) and key == "potential_names" and value:
                        row.append(value[0] if value else "")
                    else:
                        row.append("")
                else:
                    row.append("")
                    
            # Add top matches (up to 3) in separate columns
            top_matches = detailed_scores.get("top_matches", [])
            for i in range(3):  # Add up to 3 top matches
                if i < len(top_matches):
                    match = top_matches[i]
                    row.extend([match.get("name", ""), match.get("floor", ""), f"{match.get('score', 0):.4f}"])
                else:
                    row.extend(["", "", ""])
            
            writer.writerow(row)
    
    # 3. Generate a new CSV that contains all candidate scores for each image
    candidates_scores_file = os.path.join(PREDICTOR_RESULTS_DIR, f"candidate_scores_{timestamp}.csv")
    csv_files.append(candidates_scores_file)
    
    # Define component score keys to exclude
    exclude_component_keys = ['part_ratios', 'middle_token_ratios']
    
    # Find all possible component score keys across all candidates
    component_score_keys = set()
    for image_id, image_data in evaluation_results["images"].items():
        if "candidate_scores" in image_data["trained_ocr"]:
            for candidate_data in image_data["trained_ocr"]["candidate_scores"]:
                if "detailed_scores" in candidate_data:
                    # Filter out keys we want to exclude
                    keys_to_add = [k for k in candidate_data["detailed_scores"].keys() if k not in exclude_component_keys]
                    component_score_keys.update(keys_to_add)
    
    # Sort keys for consistent column order
    component_score_keys = sorted(component_score_keys)
    
    with open(candidates_scores_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image ID', 'Ground Truth', 'Candidate Name', 'Floor', 'Final Score', 'Weight Mode', 'Is Best Match', 'Is Correct'] + component_score_keys)
        
        for image_id, image_data in evaluation_results["images"].items():
            if "candidate_scores" in image_data["trained_ocr"]:
                gt_name = image_data["ground_truth"]["name"]
                
                for candidate_data in image_data["trained_ocr"]["candidate_scores"]:
                    candidate_name = candidate_data["name"]
                    candidate_floor = candidate_data["floor"]
                    final_score = candidate_data["final_score"]
                    is_best_match = candidate_data.get("is_best_match", False)
                    is_correct = gt_name == candidate_name
                    weight_mode = candidate_data.get("detailed_scores", {}).get("weight_mode", "")
                    
                    # Start with basic columns
                    row = [
                        image_id, gt_name, 
                        candidate_name, candidate_floor, 
                        f"{final_score:.4f}",
                        weight_mode,
                        "Yes" if is_best_match else "No",
                        "Yes" if is_correct else "No"
                    ]
                    
                    # Add component scores (handling only non-list or single values)
                    detailed_scores = candidate_data.get("detailed_scores", {})
                    for key in component_score_keys:
                        if key in detailed_scores:
                            value = detailed_scores[key]
                            if isinstance(value, (float, int, bool)):
                                if isinstance(value, float):
                                    row.append(f"{value:.4f}")
                                else:
                                    row.append(str(value))
                            else:
                                row.append("")
                        else:
                            row.append("")
                    
                    writer.writerow(row)
    
    # 4. NEW: Generate a CSV with weighted score components for the top candidate
    weighted_scores_file = os.path.join(PREDICTOR_RESULTS_DIR, f"weighted_scores_top_candidate_{timestamp}.csv")
    csv_files.append(weighted_scores_file)
    
    with open(weighted_scores_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header row with all component scores and their weighted values
        header = [
            'Image ID', 'Ground Truth', 'Top Candidate', 'Final Score', 'Weight Mode', 'Is Correct',
            # Original score components
            'first_last_score', 'token_set_ratio_score', 'all_parts_score', 
            'token_freq_score', 'concat_score', 'full_concat_score',
            # Weights
            'first_last_weight', 'token_set_ratio_weight', 'all_parts_weight',
            'token_freq_weight', 'concat_weight', 'full_concat_weight',
            # Weighted components (score * weight)
            'weighted_first_last', 'weighted_token_set_ratio', 'weighted_all_parts',
            'weighted_token_freq', 'weighted_concat', 'weighted_full_concat'
        ]
        writer.writerow(header)
        
        for image_id, image_data in evaluation_results["images"].items():
            gt_name = image_data["ground_truth"]["name"]
            trained_data = image_data["trained_ocr"]
            
            # Find the top candidate (best match)
            top_candidate = None
            for candidate_data in trained_data.get("candidate_scores", []):
                if candidate_data.get("is_best_match", False):
                    top_candidate = candidate_data
                    break
            
            if not top_candidate:
                continue  # Skip if no top candidate found
            
            candidate_name = top_candidate["name"]
            final_score = top_candidate["final_score"]
            is_correct = gt_name == candidate_name
            
            # Get score components from the top candidate
            component_scores = top_candidate.get("detailed_scores", {})
            weight_mode = component_scores.get("weight_mode", "unknown")
            
            # Get original score components (with defaults if missing)
            first_last = component_scores.get("first_last_score", 0.0)
            token_set = component_scores.get("token_set_ratio_score", 0.0)
            all_parts = component_scores.get("all_parts_score", 0.0)
            token_freq = component_scores.get("token_freq_score", 0.0)
            concat = component_scores.get("concat_score", 0.0)
            full_concat = component_scores.get("full_concat_score", 0.0)
            
            # Get weight values
            first_last_weight = component_scores.get("first_last_weight", 0.0)
            token_set_weight = component_scores.get("token_set_ratio_weight", 0.0)
            all_parts_weight = component_scores.get("all_parts_weight", 0.0)
            token_freq_weight = component_scores.get("token_freq_weight", 0.0)
            concat_weight = component_scores.get("concat_weight", 0.0)
            full_concat_weight = component_scores.get("full_concat_weight", 0.0)
            
            # Calculate weighted components
            weighted_first_last = first_last * first_last_weight
            weighted_token_set = token_set * token_set_weight
            weighted_all_parts = all_parts * all_parts_weight
            weighted_token_freq = token_freq * token_freq_weight
            weighted_concat = concat * concat_weight
            weighted_full_concat = full_concat * full_concat_weight
            
            # Create row with all data
            row = [
                image_id, gt_name, candidate_name, f"{final_score:.4f}", weight_mode, "Yes" if is_correct else "No",
                # Original score components
                f"{first_last:.4f}", f"{token_set:.4f}", f"{all_parts:.4f}",
                f"{token_freq:.4f}", f"{concat:.4f}", f"{full_concat:.4f}",
                # Weights
                f"{first_last_weight:.2f}", f"{token_set_weight:.2f}", f"{all_parts_weight:.2f}",
                f"{token_freq_weight:.2f}", f"{concat_weight:.2f}", f"{full_concat_weight:.2f}",
                # Weighted components
                f"{weighted_first_last:.4f}", f"{weighted_token_set:.4f}", f"{weighted_all_parts:.4f}",
                f"{weighted_token_freq:.4f}", f"{weighted_concat:.4f}", f"{weighted_full_concat:.4f}"
            ]
            writer.writerow(row)
    
    return csv_files

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
    
    # Generate detailed CSV files
    csv_files = generate_detailed_csv(evaluation_results, timestamp)
    
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
    print(f"CSV files generated: {len(csv_files)}")
    for file_path in csv_files:
        print(f"  - {os.path.basename(file_path)}")
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