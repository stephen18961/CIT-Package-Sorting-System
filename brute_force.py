import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import cv2
import sys
from itertools import product
from tqdm import tqdm
import json

# Add the parent directory to sys.path to import modules from your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules from your existing code
from ocr_module import init_ocr_pipeline, run_ocr_on_frame
from predictor import approximate_receiver_name, noise_words
from extensions import db, app
from models import Student, Staff

class OCRParameterOptimizer:
    def __init__(self, ground_truth_path, images_folder, results_folder, use_gpu=False):
        """
        Initialize the OCR Parameter Optimizer
        
        Args:
            ground_truth_path: Path to ground truth JSON file
            images_folder: Path to folder containing test images
            results_folder: Path to save results
            use_gpu: Whether to use GPU for processing
        """
        self.ground_truth_path = ground_truth_path
        self.images_folder = images_folder
        self.results_folder = results_folder
        self.use_gpu = use_gpu
        
        # Create results folder if it doesn't exist
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Load ground truth
        self.ground_truth_db = self._load_ground_truth()
        
        # Load candidate names from the database
        with app.app_context():
            self.candidate_names = self._load_candidates_from_db()
        
        print(f"Initialized OCR Parameter Optimizer")
        print(f"Ground truth database has {len(self.ground_truth_db)} entries")
        print(f"Using {len(self.candidate_names)} candidate names")
        print(f"GPU usage: {'Enabled' if self.use_gpu else 'Disabled'}")
    
    def _load_ground_truth(self):
        """
        Load existing ground truth database from JSON file
        """
        try:
            if os.path.exists(self.ground_truth_path):
                with open(self.ground_truth_path, 'r') as f:
                    ground_truth_json = json.load(f)
                
                # Convert JSON to DataFrame format
                records = []
                for image_filename, data in ground_truth_json.items():
                    # Add .jpg extension if not already present
                    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_filename = f"{image_filename}.jpg"
                        
                    records.append({
                        'image_filename': image_filename,
                        'ground_truth_name': data['name'],
                        'ground_truth_floor': data['floor'],
                        'source': data.get('source', 'unknown'),
                        'verified_date': datetime.now().strftime("%Y-%m-%d")  # Add current date as verification date
                    })
                
                df = pd.DataFrame(records)
                print(f"Loaded ground truth with {len(df)} entries from {self.ground_truth_path}")
                return df
            else:
                print(f"Ground truth file not found: {self.ground_truth_path}")
                return pd.DataFrame(columns=['image_filename', 'ground_truth_name', 'ground_truth_floor', 'source', 'verified_date'])
        except Exception as e:
            print(f"Error loading ground truth database: {e}")
            return pd.DataFrame(columns=['image_filename', 'ground_truth_name', 'ground_truth_floor', 'source', 'verified_date'])
    
    def _load_candidates_from_db(self):
        """
        Load candidate names from the database
        """
        try:
            # Get students from database
            students = [(s.name, str(s.floor)) for s in Student.query.all()]
            
            # Get staff from database
            staff = [(s.name, str(s.floor)) for s in Staff.query.all()]
            
            # Combine lists
            all_candidates = students + staff
            
            print(f"Loaded {len(students)} students and {len(staff)} staff members from database")
            return all_candidates
        except Exception as e:
            print(f"Error loading names from database: {e}")
            print("Falling back to dummy data")
            # Fallback to dummy data if database access fails
            return [
                ("John Smith", "3"),
                ("Jane Doe", "5"),
                ("Robert Johnson", "2"),
                ("Emily Wilson", "4"),
                ("Michael Brown", "1"),
                # Add more dummy names as needed
            ]
    
    def load_test_images(self):
        """
        Load test images that have ground truth data
        """
        image_filenames = self.ground_truth_db['image_filename'].unique()
        image_paths = []
        
        for filename in image_filenames:
            image_path = os.path.join(self.images_folder, filename)
            if os.path.exists(image_path):
                image_paths.append(image_path)
            else:
                print(f"Warning: Image {filename} not found in {self.images_folder}")
        
        print(f"Found {len(image_paths)} images with ground truth data")
        return image_paths
    
    def run_ocr_test(self, pipeline, image_path, target_width, match_threshold):
        """
        Run OCR on a single image with specified parameters
        
        Args:
            pipeline: Initialized keras-OCR pipeline
            image_path: Path to the image file
            target_width: Width to resize images for OCR
            match_threshold: Threshold for name matching
            
        Returns:
            Dictionary with test results
        """
        image_filename = os.path.basename(image_path)
        
        # Get ground truth for this image
        ground_truth_row = self.ground_truth_db[self.ground_truth_db['image_filename'] == image_filename]
        if ground_truth_row.empty:
            return None
        
        ground_truth_name = ground_truth_row['ground_truth_name'].iloc[0]
        ground_truth_floor = ground_truth_row['ground_truth_floor'].iloc[0]
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        try:
            # Create a temporary folder for OCR results
            temp_folder = os.path.join(self.results_folder, f"temp_{int(time.time())}")
            os.makedirs(temp_folder, exist_ok=True)
            
            # Run OCR
            text_path, _, recognized_lines = run_ocr_on_frame(
                pipeline=pipeline,
                frame=frame,
                results_folder=temp_folder,
                target_width=target_width
            )
            
            # Combine OCR text
            ocr_text_combined = "\n".join(recognized_lines)
            
            # Run rule-based predictor
            best_match, score = approximate_receiver_name(
                ocr_text_combined,
                self.candidate_names,
                noise_words,
                threshold=0.8,
                match_threshold=match_threshold
            )
            
            # Clean up temporary folder
            try:
                import shutil
                shutil.rmtree(temp_folder)
            except:
                pass
            
            # Prepare result
            result = {
                'image_filename': image_filename,
                'ocr_text': ocr_text_combined,
                'predicted_name': best_match[0] if best_match else "No match",
                'predicted_floor': best_match[1] if best_match else "N/A",
                'confidence_score': round(score, 4),
                'match_found': best_match is not None,
                'ground_truth_name': ground_truth_name,
                'ground_truth_floor': ground_truth_floor,
                'is_correct': (best_match[0] if best_match else "No match") == ground_truth_name,
                'width': target_width,
                'threshold': match_threshold
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing image {image_filename} with width={target_width}, threshold={match_threshold}: {e}")
            return {
                'image_filename': image_filename,
                'ocr_text': "ERROR",
                'predicted_name': "Error during processing",
                'predicted_floor': "N/A",
                'confidence_score': 0,
                'match_found': False,
                'ground_truth_name': ground_truth_name,
                'ground_truth_floor': ground_truth_floor,
                'is_correct': False,
                'width': target_width,
                'threshold': match_threshold,
                'error': str(e)
            }
    
    def grid_search(self, model_path, width_range, threshold_range):
        """
        Perform grid search to find optimal parameters
        
        Args:
            model_path: Path to trained model
            width_range: List of width values to test
            threshold_range: List of threshold values to test
            
        Returns:
            DataFrame with results and best parameters
        """
        print("\nInitializing OCR pipeline...")
        pipeline = init_ocr_pipeline(
            custom_recognizer_path=model_path,
            custom_detector_path=None,
            use_gpu=self.use_gpu
        )
        
        # Load test images
        image_paths = self.load_test_images()
        if not image_paths:
            print("No test images found with ground truth data!")
            return None, None
        
        # Generate parameter combinations
        param_combinations = list(product(width_range, threshold_range))
        print(f"Testing {len(param_combinations)} parameter combinations on {len(image_paths)} images")
        
        # Create a DataFrame to store results
        results = []
        
        # Loop through all parameter combinations
        total_iterations = len(param_combinations) * len(image_paths)
        with tqdm(total=total_iterations, desc="Testing parameters") as pbar:
            for width, threshold in param_combinations:
                param_results = []
                
                for image_path in image_paths:
                    result = self.run_ocr_test(pipeline, image_path, width, threshold)
                    if result:
                        param_results.append(result)
                    pbar.update(1)
                
                # Calculate accuracy for this parameter combination
                accuracy = sum(r['is_correct'] for r in param_results) / len(param_results) if param_results else 0
                
                # Store summary results
                summary = {
                    'width': width,
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'total_images': len(param_results),
                    'correct_predictions': sum(r['is_correct'] for r in param_results),
                    'match_found_rate': sum(r['match_found'] for r in param_results) / len(param_results) if param_results else 0
                }
                
                results.append(summary)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best parameters
        if not results_df.empty:
            best_params = results_df.loc[results_df['accuracy'].idxmax()]
            return results_df, best_params
        else:
            print("No results generated!")
            return None, None
    
    def generate_report(self, results_df, best_params):
        """
        Generate and save optimization report
        
        Args:
            results_df: DataFrame with grid search results
            best_params: Series with best parameters
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_path = os.path.join(self.results_folder, f"parameter_optimization_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Save best parameters
        best_params_path = os.path.join(self.results_folder, f"best_parameters_{timestamp}.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_width': int(best_params['width']),
                'best_threshold': float(best_params['threshold']),
                'accuracy': float(best_params['accuracy']),
                'timestamp': timestamp
            }, f, indent=4)
        
        # Create appropriate visualizations based on the parameter ranges
        report_paths = {
            'results_path': results_path,
            'best_params_path': best_params_path
        }
        
        # Get unique parameter values
        unique_widths = results_df['width'].unique()
        unique_thresholds = results_df['threshold'].unique()
        
        # Determine if we should create a heatmap (both params have multiple values)
        if len(unique_widths) > 1 and len(unique_thresholds) > 1:
            # Create heatmap visualization
            plt.figure(figsize=(12, 10))
            
            # Pivot data for heatmap
            pivot_data = results_df.pivot(index="width", columns="threshold", values="accuracy")
            
            # Create heatmap
            heatmap = plt.imshow(pivot_data, cmap='viridis', interpolation='nearest')
            plt.colorbar(heatmap, label='Accuracy')
            
            # Add labels
            plt.title('Parameter Optimization Results: OCR Width vs Match Threshold', fontsize=14)
            plt.xlabel('Match Threshold', fontsize=12)
            plt.ylabel('OCR Width', fontsize=12)
            
            # Add threshold values as x-tick labels
            thresholds = sorted(unique_thresholds)
            plt.xticks(range(len(thresholds)), [f"{t:.2f}" for t in thresholds], rotation=45)
            
            # Add width values as y-tick labels
            widths = sorted(unique_widths)
            plt.yticks(range(len(widths)), widths)
            
            # Mark best parameters
            best_x = list(thresholds).index(best_params['threshold'])
            best_y = list(widths).index(best_params['width'])
            plt.plot(best_x, best_y, 'r*', markersize=15, label=f"Best: Width={best_params['width']}, Threshold={best_params['threshold']:.2f}")
            plt.legend()
            
            # Save heatmap
            heatmap_path = os.path.join(self.results_folder, f"parameter_heatmap_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=150)
            plt.close()
            report_paths['heatmap_path'] = heatmap_path
        
        # Create line plots based on what parameters were varied
        if len(unique_widths) > 1 or len(unique_thresholds) > 1:
            plt.figure(figsize=(10, 6))
            
            if len(unique_widths) > 1:
                # If testing multiple widths, create width plot
                if len(unique_thresholds) == 1:
                    # Single threshold value
                    plt.plot(results_df['width'], results_df['accuracy'], marker='o', linewidth=2)
                    plt.title(f'Accuracy vs OCR Width (Threshold={unique_thresholds[0]:.2f})', fontsize=14)
                else:
                    # Multiple threshold values, show averaged results
                    width_data = results_df.groupby('width')['accuracy'].mean().reset_index()
                    plt.plot(width_data['width'], width_data['accuracy'], marker='o', linewidth=2)
                    plt.title('Average Accuracy vs OCR Width', fontsize=14)
                
                plt.xlabel('OCR Width', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Mark best width
                best_width = best_params['width']
                best_accuracy = results_df[results_df['width'] == best_width]['accuracy'].max()
                plt.plot(best_width, best_accuracy, 'r*', markersize=15, 
                         label=f"Best: Width={best_width}")
                plt.legend()
                
            elif len(unique_thresholds) > 1:
                # If testing multiple thresholds, create threshold plot
                if len(unique_widths) == 1:
                    # Single width value
                    plt.plot(results_df['threshold'], results_df['accuracy'], marker='o', linewidth=2, color='green')
                    plt.title(f'Accuracy vs Match Threshold (Width={unique_widths[0]})', fontsize=14)
                else:
                    # Multiple width values, show averaged results
                    threshold_data = results_df.groupby('threshold')['accuracy'].mean().reset_index()
                    plt.plot(threshold_data['threshold'], threshold_data['accuracy'], marker='o', linewidth=2, color='green')
                    plt.title('Average Accuracy vs Match Threshold', fontsize=14)
                
                plt.xlabel('Match Threshold', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Mark best threshold
                best_threshold = best_params['threshold']
                best_accuracy = results_df[results_df['threshold'] == best_threshold]['accuracy'].max()
                plt.plot(best_threshold, best_accuracy, 'r*', markersize=15, 
                         label=f"Best: Threshold={best_threshold:.2f}")
                plt.legend()
            
            # Save plot
            plot_path = os.path.join(self.results_folder, f"parameter_plot_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            report_paths['plot_path'] = plot_path
        
        # Print summary
        print("\n" + "="*60)
        print("OCR PARAMETER OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Best Width: {int(best_params['width'])}")
        print(f"Best Match Threshold: {best_params['threshold']:.4f}")
        print(f"Accuracy with best parameters: {best_params['accuracy']*100:.2f}%")
        print(f"Images tested: {best_params['total_images']}")
        print(f"Correct predictions: {best_params['correct_predictions']}/{best_params['total_images']}")
        print(f"Match found rate: {best_params['match_found_rate']*100:.2f}%")
        print(f"Results saved to: {results_path}")
        print(f"Best parameters saved to: {best_params_path}")
        
        for path_name, path in report_paths.items():
            if path_name not in ['results_path', 'best_params_path']:
                print(f"{path_name.replace('_path', '').capitalize()} saved to: {path}")
                
        print("="*60)
        
        return report_paths


def main():
    parser = argparse.ArgumentParser(description='OCR Parameter Optimization Tool')
    parser.add_argument('--ground-truth', default='ocr_accuracy_results/ground_truth.json', 
                        help='Path to ground truth JSON file')
    parser.add_argument('--images', default='test_data', 
                        help='Path to test images folder')
    parser.add_argument('--results', default='parameter_optimization_results', 
                        help='Path to results folder')
    parser.add_argument('--model', default='models/recognizer_with_augment.h5', 
                        help='Path to trained recognizer model')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU for processing')
    
    # Width parameters
    width_group = parser.add_argument_group('Width parameters')
    width_group.add_argument('--width', type=int,
                        help='Single OCR width value to test')
    width_group.add_argument('--min-width', type=int, default=560, 
                        help='Minimum OCR preprocess width')
    width_group.add_argument('--max-width', type=int, default=800, 
                        help='Maximum OCR preprocess width')
    width_group.add_argument('--width-step', type=int, default=80, 
                        help='Step size for OCR width')
    
    # Threshold parameters
    threshold_group = parser.add_argument_group('Threshold parameters')
    threshold_group.add_argument('--threshold', type=float,
                        help='Single match threshold value to test')
    threshold_group.add_argument('--min-threshold', type=float, default=0.36, 
                        help='Minimum match threshold')
    threshold_group.add_argument('--max-threshold', type=float, default=0.50, 
                        help='Maximum match threshold')
    threshold_group.add_argument('--threshold-step', type=float, default=0.02, 
                        help='Step size for match threshold')
    
    args = parser.parse_args()
    
    # Generate width range
    if args.width is not None:
        # If a single width is specified, use only that value
        width_range = [args.width]
    else:
        # Otherwise generate range based on min, max, step
        width_range = list(range(args.min_width, args.max_width + args.width_step, args.width_step))
    
    # Generate threshold range
    if args.threshold is not None:
        # If a single threshold is specified, use only that value
        threshold_range = [args.threshold]
    else:
        # Otherwise generate range based on min, max, step
        threshold_range = [round(t, 2) for t in np.arange(args.min_threshold, args.max_threshold + args.threshold_step, args.threshold_step)]
    
    print(f"Width range: {width_range}")
    print(f"Threshold range: {threshold_range}")
    
    # Initialize optimizer
    optimizer = OCRParameterOptimizer(
        ground_truth_path=args.ground_truth,
        images_folder=args.images,
        results_folder=args.results,
        use_gpu=args.gpu
    )
    
    # Run grid search
    results_df, best_params = optimizer.grid_search(
        model_path=args.model,
        width_range=width_range,
        threshold_range=threshold_range
    )
    
    # Generate report
    if results_df is not None and best_params is not None:
        report_paths = optimizer.generate_report(results_df, best_params)
        
        print("\nOptimization completed!")
        print(f"Best Width: {int(best_params['width'])}")
        print(f"Best Match Threshold: {best_params['threshold']:.4f}")
        print(f"Accuracy: {best_params['accuracy']*100:.2f}%")
    else:
        print("Optimization failed - no results generated.")

if __name__ == "__main__":
    main()