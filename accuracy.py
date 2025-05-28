import os
import time
import cv2
import keras_ocr
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import sys
import matplotlib

# Add the parent directory to sys.path to import modules from your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules from your existing code
from ocr_module import preprocess_image, init_ocr_pipeline, run_ocr_on_frame
from predictor import approximate_receiver_name, noise_words
from extensions import db, app
from models import Student, Staff

class OCRAccuracyTester:
    def __init__(self, images_folder, results_folder, use_gpu=True, target_width=560, match_threshold=0.36, interactive=True):
        """
        Initialize the OCR Accuracy Tester
        
        Args:
            images_folder: Path to folder containing test images
            results_folder: Path to save results
            use_gpu: Whether to use GPU for processing
            target_width: Width to resize images for OCR
            match_threshold: Threshold for name matching
            interactive: Whether to use interactive visualization
        """
        self.images_folder = images_folder
        self.results_folder = results_folder
        self.use_gpu = use_gpu
        self.interactive = interactive
        
        # Check if we're in an environment where we can display plots
        if self.interactive:
            try:
                # Try to set a non-interactive backend if we're in a non-interactive environment
                if not plt.isinteractive():
                    matplotlib.use('Agg')  # Use non-interactive backend
                    self.interactive = False
                    print("Non-interactive environment detected. Disabling interactive visualizations.")
            except Exception as e:
                print(f"Warning: Error checking interactive display: {e}")
                self.interactive = False
        
        # Create results folder if it doesn't exist
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, "base_model"), exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, "trained_model"), exist_ok=True)
        
        # Create ground truth folder
        self.ground_truth_folder = os.path.join(self.results_folder, "ground_truth")
        os.makedirs(self.ground_truth_folder, exist_ok=True)
        
        # Create folder for saved images when in non-interactive mode
        self.visualization_folder = os.path.join(self.results_folder, "visualizations")
        if not self.interactive:
            os.makedirs(self.visualization_folder, exist_ok=True)
        
        # Parameters
        self.ocr_preprocess_width = target_width
        self.match_threshold = match_threshold
        
        # Test results
        self.results = []
        
        # Ground truth database
        self.ground_truth_file = os.path.join(self.ground_truth_folder, "verified_ground_truth.csv")
        self.ground_truth_db = self._load_ground_truth()
        
        # Load candidate names from the database
        with app.app_context():
            self.candidate_names = self._load_candidates_from_db()
        
        print(f"Initialized OCR Accuracy Tester with {len(self.candidate_names)} candidates")
        print(f"Using OCR width: {self.ocr_preprocess_width}, Match threshold: {self.match_threshold}")
        print(f"GPU usage: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"Interactive mode: {'Enabled' if self.interactive else 'Disabled'}")
        print(f"Ground truth database has {len(self.ground_truth_db)} entries")
    
    def _load_ground_truth(self):
        """
        Load existing ground truth database if it exists
        """
        if os.path.exists(self.ground_truth_file):
            try:
                return pd.read_csv(self.ground_truth_file)
            except Exception as e:
                print(f"Error loading ground truth database: {e}")
                return pd.DataFrame(columns=['image_filename', 'ground_truth_name', 'ground_truth_floor', 'verified_date'])
        else:
            return pd.DataFrame(columns=['image_filename', 'ground_truth_name', 'ground_truth_floor', 'verified_date'])
    
    def _save_ground_truth(self):
        """
        Save ground truth database to file
        """
        try:
            self.ground_truth_db.to_csv(self.ground_truth_file, index=False)
            print(f"Ground truth database saved with {len(self.ground_truth_db)} entries")
        except Exception as e:
            print(f"Error saving ground truth database: {e}")
    
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
    
    def load_images(self):
        """
        Load all images from the specified folder
        """
        image_paths = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for filename in os.listdir(self.images_folder):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_paths.append(os.path.join(self.images_folder, filename))
        
        print(f"Found {len(image_paths)} images in {self.images_folder}")
        return sorted(image_paths)  # Sort for consistent ordering
    
    def run_test(self, pipeline, image_path, model_type):
        """
        Run OCR and predictor on a single image
        
        Args:
            pipeline: Initialized keras-OCR pipeline
            image_path: Path to the image file
            model_type: String identifier for the model being tested
            
        Returns:
            Dictionary with test results
        """
        image_filename = os.path.basename(image_path)
        print(f"\nTesting {model_type} on {image_filename}")
        
        # Check if we have ground truth for this image
        ground_truth = None
        ground_truth_row = self.ground_truth_db[self.ground_truth_db['image_filename'] == image_filename]
        if not ground_truth_row.empty:
            ground_truth = (ground_truth_row['ground_truth_name'].iloc[0], 
                           ground_truth_row['ground_truth_floor'].iloc[0])
            print(f"Found ground truth: {ground_truth[0]} (Floor: {ground_truth[1]})")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        try:
            # Run OCR
            text_path, annotated_image_path, recognized_lines = run_ocr_on_frame(
                pipeline=pipeline,
                frame=frame,
                results_folder=os.path.join(self.results_folder, model_type),
                target_width=self.ocr_preprocess_width
            )
            
            # Combine OCR text
            ocr_text_combined = "\n".join(recognized_lines)
            
            # Run rule-based predictor
            best_match, score = approximate_receiver_name(
                ocr_text_combined,
                self.candidate_names,
                noise_words,
                threshold=0.8,
                match_threshold=self.match_threshold
            )
            
            # Prepare result
            result = {
                'image_filename': image_filename,
                'model_type': model_type,
                'ocr_text': ocr_text_combined,
                'predicted_name': best_match[0] if best_match else "No match",
                'predicted_floor': best_match[1] if best_match else "N/A",
                'confidence_score': round(score, 4),
                'annotated_image_path': annotated_image_path,
                'original_image_path': image_path,
                'match_found': best_match is not None,
                'ground_truth_name': ground_truth[0] if ground_truth else None,
                'ground_truth_floor': ground_truth[1] if ground_truth else None,
                'has_ground_truth': ground_truth is not None
            }
            
            print(f"Prediction: {result['predicted_name']} (Floor: {result['predicted_floor']}), Score: {result['confidence_score']}")
            if ground_truth:
                print(f"Ground Truth: {ground_truth[0]} (Floor: {ground_truth[1]})")
            
            return result
            
        except Exception as e:
            print(f"Error processing image {image_filename} with {model_type}: {e}")
            return {
                'image_filename': image_filename,
                'model_type': model_type,
                'ocr_text': "ERROR",
                'predicted_name': "Error during processing",
                'predicted_floor': "N/A",
                'confidence_score': 0,
                'annotated_image_path': None,
                'original_image_path': image_path,
                'match_found': False,
                'ground_truth_name': None,
                'ground_truth_floor': None,
                'has_ground_truth': False,
                'error': str(e)
            }
    
    def test_all_images(self, base_model_pipeline, trained_model_pipeline):
        """
        Test all images with both models
        
        Args:
            base_model_pipeline: Base keras-OCR pipeline
            trained_model_pipeline: Trained keras-OCR pipeline
        """
        image_paths = self.load_images()
        
        for idx, image_path in enumerate(image_paths):
            print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Test with base model
            base_result = self.run_test(base_model_pipeline, image_path, "base_model")
            if base_result:
                self.results.append(base_result)
            
            # Test with trained model
            trained_result = self.run_test(trained_model_pipeline, image_path, "trained_model")
            if trained_result:
                self.results.append(trained_result)
    
    def _save_visualizations(self, image_filename, original_path, annotated_paths):
        """
        Save visualizations to disk in non-interactive mode
        
        Args:
            image_filename: Base name of the image file
            original_path: Path to original image
            annotated_paths: Dictionary of model_type -> annotated image path
            
        Returns:
            Path to saved visualization
        """
        try:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(18, 8))
            
            # First add the original image
            ax1 = fig.add_subplot(1, 3, 1)
            original_img = cv2.imread(original_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            ax1.imshow(original_img)
            ax1.set_title(f"Original: {image_filename}")
            ax1.axis('off')
            
            # Add annotated images
            for i, (model_type, path) in enumerate(annotated_paths.items(), start=2):
                if path and os.path.exists(path):
                    ax = fig.add_subplot(1, 3, i)
                    annotated_img = cv2.imread(path)
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    ax.imshow(annotated_img)
                    ax.set_title(f"{model_type} OCR")
                    ax.axis('off')
            
            # Save the figure
            vis_file = os.path.join(self.visualization_folder, f"{os.path.splitext(image_filename)[0]}_comparison.png")
            plt.tight_layout()
            plt.savefig(vis_file, dpi=150)
            plt.close(fig)
            
            return vis_file
        except Exception as e:
            print(f"Error saving visualizations: {e}")
            return None
    
    def interactive_verification(self):
        """
        Interactive verification of results, with fallback for non-interactive environments
        """
        df = pd.DataFrame(self.results)
        
        # Group by image to compare models
        for image_filename, group in df.groupby('image_filename'):
            if self.interactive:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
            
            # Display the original image
            original_path = group['original_image_path'].iloc[0]
            
            try:
                # Check if we need to save visualizations instead of displaying them
                annotated_paths = {}
                for _, row in group.iterrows():
                    if row['annotated_image_path']:
                        annotated_paths[row['model_type']] = row['annotated_image_path']
                
                if self.interactive:
                    # Show original image
                    plt.figure(figsize=(10, 8))
                    img = cv2.imread(original_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.title(f"Original Image: {image_filename}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show(block=False)
                    
                    # Show annotated images side by side if available
                    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                    
                    for i, model_type in enumerate(['base_model', 'trained_model']):
                        model_result = group[group['model_type'] == model_type]
                        if not model_result.empty and model_result['annotated_image_path'].iloc[0]:
                            try:
                                annotated_img = cv2.imread(model_result['annotated_image_path'].iloc[0])
                                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                                axes[i].imshow(annotated_img)
                                axes[i].set_title(f"{model_type} OCR Result")
                                axes[i].axis('off')
                            except Exception as e:
                                axes[i].text(0.5, 0.5, f"Error loading annotated image: {e}", 
                                           ha='center', va='center', transform=axes[i].transAxes)
                    
                    plt.tight_layout()
                    plt.show(block=False)
                else:
                    # In non-interactive mode, save visualizations to disk
                    vis_path = self._save_visualizations(image_filename, original_path, annotated_paths)
                    if vis_path:
                        print(f"Saved visualization to: {vis_path}")
                
                # Check if we already have ground truth
                has_ground_truth = False
                ground_truth_name = None
                ground_truth_floor = None
                
                # Check if ground truth exists in dataframe
                if 'has_ground_truth' in group.columns and group['has_ground_truth'].any():
                    ground_truth_row = group[group['has_ground_truth']].iloc[0]
                    has_ground_truth = True
                    ground_truth_name = ground_truth_row['ground_truth_name']
                    ground_truth_floor = ground_truth_row['ground_truth_floor']
                    print(f"\nGround Truth: {ground_truth_name} (Floor: {ground_truth_floor})")
                
                # Display results for comparison
                comparison_data = []
                for _, row in group.iterrows():
                    comparison_data.append([
                        row['model_type'],
                        row['predicted_name'],
                        row['predicted_floor'],
                        row['confidence_score'],
                        'Yes' if row['match_found'] else 'No'
                    ])
                
                print(tabulate(comparison_data, 
                      headers=['Model', 'Predicted Name', 'Floor', 'Confidence', 'Match Found'],
                      tablefmt='grid'))
                
                # Display OCR text from both models
                print("\nBase Model OCR Text:")
                base_text = group[group['model_type'] == 'base_model']['ocr_text'].iloc[0]
                print(base_text[:500] + ('...' if len(base_text) > 500 else ''))
                
                print("\nTrained Model OCR Text:")
                trained_text = group[group['model_type'] == 'trained_model']['ocr_text'].iloc[0]
                print(trained_text[:500] + ('...' if len(trained_text) > 500 else ''))
                
                # Ask for ground truth if not already available
                if not has_ground_truth:
                    print("\nNo ground truth available for this image.")
                    set_ground_truth = input("Do you want to set ground truth? (y/n): ").lower().strip()
                    
                    if set_ground_truth == 'y':
                        # Ask which model prediction to use as ground truth or enter custom
                        print("\nSelect ground truth source:")
                        print("1. Use base model prediction")
                        print("2. Use trained model prediction")
                        print("3. Enter custom ground truth")
                        
                        choice = input("Enter choice (1-3): ").strip()
                        
                        if choice == '1':
                            # Use base model prediction
                            base_row = group[group['model_type'] == 'base_model'].iloc[0]
                            ground_truth_name = base_row['predicted_name']
                            ground_truth_floor = base_row['predicted_floor']
                        elif choice == '2':
                            # Use trained model prediction
                            trained_row = group[group['model_type'] == 'trained_model'].iloc[0]
                            ground_truth_name = trained_row['predicted_name']
                            ground_truth_floor = trained_row['predicted_floor']
                        elif choice == '3':
                            # Enter custom ground truth
                            ground_truth_name = input("Enter ground truth name: ").strip()
                            ground_truth_floor = input("Enter ground truth floor: ").strip()
                        else:
                            print("Invalid choice. Skipping ground truth setting.")
                            ground_truth_name = None
                            ground_truth_floor = None
                        
                        # Save ground truth if provided
                        if ground_truth_name and ground_truth_name != "No match":
                            # Update ground truth database
                            new_entry = pd.DataFrame({
                                'image_filename': [image_filename],
                                'ground_truth_name': [ground_truth_name],
                                'ground_truth_floor': [ground_truth_floor],
                                'verified_date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                            })
                            
                            # Remove existing entry if exists
                            self.ground_truth_db = self.ground_truth_db[
                                self.ground_truth_db['image_filename'] != image_filename]
                            
                            # Add new entry
                            self.ground_truth_db = pd.concat([self.ground_truth_db, new_entry], ignore_index=True)
                            
                            # Save to file
                            self._save_ground_truth()
                            
                            print(f"Ground truth set: {ground_truth_name} (Floor: {ground_truth_floor})")
                            
                            # Update dataframe with ground truth
                            for idx in df.index[(df['image_filename'] == image_filename)].tolist():
                                df.at[idx, 'ground_truth_name'] = ground_truth_name
                                df.at[idx, 'ground_truth_floor'] = ground_truth_floor
                                df.at[idx, 'has_ground_truth'] = True
                
                # Ask for verification for each model
                for model_type in ['base_model', 'trained_model']:
                    model_result = group[group['model_type'] == model_type]
                    if not model_result.empty:
                        predicted_name = model_result['predicted_name'].iloc[0]
                        
                        correct = input(f"\nIs the {model_type} prediction ({predicted_name}) correct? (y/n): ").lower().strip()
                        while correct not in ['y', 'n']:
                            correct = input("Please enter 'y' or 'n': ").lower().strip()
                        
                        # Update the dataframe with verification result
                        idx = df.index[(df['image_filename'] == image_filename) & 
                                     (df['model_type'] == model_type)].tolist()[0]
                        df.at[idx, 'is_correct'] = (correct == 'y')
                        
                        # If correct and no ground truth set, use prediction as ground truth
                        if correct == 'y' and not has_ground_truth and not ground_truth_name:
                            ground_truth_name = predicted_name
                            ground_truth_floor = model_result['predicted_floor'].iloc[0]
                            
                            # Update ground truth database
                            new_entry = pd.DataFrame({
                                'image_filename': [image_filename],
                                'ground_truth_name': [ground_truth_name],
                                'ground_truth_floor': [ground_truth_floor],
                                'verified_date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                            })
                            
                            # Remove existing entry if exists
                            self.ground_truth_db = self.ground_truth_db[
                                self.ground_truth_db['image_filename'] != image_filename]
                            
                            # Add new entry
                            self.ground_truth_db = pd.concat([self.ground_truth_db, new_entry], ignore_index=True)
                            
                            # Save to file
                            self._save_ground_truth()
                            
                            print(f"Using correct prediction as ground truth: {ground_truth_name} (Floor: {ground_truth_floor})")
                            
                            # Update all entries for this image with the ground truth
                            for gidx in df.index[(df['image_filename'] == image_filename)].tolist():
                                df.at[gidx, 'ground_truth_name'] = ground_truth_name
                                df.at[gidx, 'ground_truth_floor'] = ground_truth_floor
                                df.at[gidx, 'has_ground_truth'] = True
                
                if self.interactive:
                    plt.close('all')  # Close all figures before moving to next image
                
                input("\nPress Enter to continue to the next image...")
                
            except Exception as e:
                print(f"Error displaying results for {image_filename}: {e}")
                input("Press Enter to continue...")
        
        return df
    
    def generate_report(self, verified_results):
        """
        Generate an accuracy report
        
        Args:
            verified_results: DataFrame with verification results
        """
        # Calculate overall accuracy
        base_model_acc = verified_results[verified_results['model_type'] == 'base_model']['is_correct'].mean()
        trained_model_acc = verified_results[verified_results['model_type'] == 'trained_model']['is_correct'].mean()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report
        report = {
            'timestamp': timestamp,
            'total_images': len(verified_results['image_filename'].unique()),
            'base_model_accuracy': round(base_model_acc * 100, 2),
            'trained_model_accuracy': round(trained_model_acc * 100, 2),
            'improvement': round((trained_model_acc - base_model_acc) * 100, 2),
            'detailed_results': verified_results,
            'ground_truth_count': len(self.ground_truth_db)
        }
        
        # Save report
        report_file = os.path.join(self.results_folder, f"accuracy_report_{timestamp}.csv")
        verified_results.to_csv(report_file, index=False)
        
        # Create detailed report with image filenames and predictions
        report_df = pd.DataFrame({
            'Image': verified_results['image_filename'],
            'Model': verified_results['model_type'],
            'Predicted Name': verified_results['predicted_name'],
            'Floor': verified_results['predicted_floor'],
            'Confidence': verified_results['confidence_score'],
            'Correct': verified_results['is_correct'],
            'Ground Truth Name': verified_results['ground_truth_name'],
            'Ground Truth Floor': verified_results['ground_truth_floor'],
            'Has Ground Truth': verified_results['has_ground_truth']
        })
        
        # Save detailed report as CSV
        detailed_report_file = os.path.join(self.results_folder, f"detailed_report_{timestamp}.csv")
        report_df.to_csv(detailed_report_file, index=False)
        
        # Also save the ground truth database (redundant but for safety)
        ground_truth_report_file = os.path.join(self.ground_truth_folder, f"ground_truth_{timestamp}.csv")
        self.ground_truth_db.to_csv(ground_truth_report_file, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("OCR ACCURACY TEST RESULTS")
        print("="*60)
        print(f"Total images tested: {report['total_images']}")
        print(f"Base Model Accuracy: {report['base_model_accuracy']}%")
        print(f"Trained Model Accuracy: {report['trained_model_accuracy']}%")
        print(f"Improvement: {report['improvement']}%")
        print(f"Ground Truth database entries: {report['ground_truth_count']}")
        print(f"Detailed results saved to: {report_file}")
        print(f"Image-by-image analysis saved to: {detailed_report_file}")
        print(f"Ground truth database saved to: {ground_truth_report_file}")
        print("="*60)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='OCR Accuracy Testing Tool')
    parser.add_argument('--images', default='test_data', help='Path to test images folder')
    parser.add_argument('--results', default='accuracy_test_results', help='Path to results folder')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('--width', type=int, default=560, help='OCR preprocess width')
    parser.add_argument('--threshold', type=float, default=0.36, help='Match threshold')
    parser.add_argument('--trained-model', default='models/recognizer_with_augment.h5', 
                        help='Path to trained recognizer model')
    parser.add_argument('--non-interactive', action='store_true', 
                        help='Run in non-interactive mode (no plot display)')
    
    args = parser.parse_args()
    
    # Set backend to non-interactive if specified
    if args.non_interactive:
        matplotlib.use('Agg')
    
    # Initialize tester
    tester = OCRAccuracyTester(
        images_folder=args.images,
        results_folder=args.results,
        use_gpu=args.gpu,
        target_width=args.width,
        match_threshold=args.threshold,
        interactive=not args.non_interactive
    )
    
    print("\nInitializing base model...")
    base_model_pipeline = init_ocr_pipeline(
        custom_recognizer_path=None,
        custom_detector_path=None,
        use_gpu=args.gpu
    )
    
    print("\nInitializing trained model...")
    trained_model_pipeline = init_ocr_pipeline(
        custom_recognizer_path=args.trained_model,
        custom_detector_path=None,
        use_gpu=args.gpu
    )
    
    print("\nStarting accuracy test...")
    tester.test_all_images(base_model_pipeline, trained_model_pipeline)
    
    print("\nStarting interactive verification...")
    verified_results = tester.interactive_verification()
    
    print("\nGenerating accuracy report...")
    report = tester.generate_report(verified_results)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()