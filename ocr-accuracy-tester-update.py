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
    def __init__(self, images_folder, results_folder, use_gpu, target_width, match_threshold, interactive):
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
        
        if self.interactive:
            try:
                if not plt.isinteractive():
                    matplotlib.use('Agg')
                    self.interactive = False
                    print("Non-interactive environment detected. Disabling interactive visualizations.")
            except Exception as e:
                print(f"Warning: Error checking interactive display: {e}")
                self.interactive = False
        
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, "base_model"), exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, "trained_model"), exist_ok=True)
        
        self.ground_truth_folder = 'accuracy_test_results/ground_truth'
        os.makedirs(self.ground_truth_folder, exist_ok=True)
        
        self.visualization_folder = os.path.join(self.results_folder, "visualizations")
        if not self.interactive:
            os.makedirs(self.visualization_folder, exist_ok=True)
        
        self.ocr_preprocess_width = target_width
        self.match_threshold = match_threshold
        
        self.results = []
        
        self.ground_truth_file = 'accuracy_test_results/ground_truth/verified_ground_truth.csv'
        self.ground_truth_db = self._load_ground_truth()
        
        with app.app_context():
            self.candidate_names = self._load_candidates_from_db()
        
        print(f"Initialized OCR Accuracy Tester with {len(self.candidate_names)} candidates")
        print(f"Using OCR width: {self.ocr_preprocess_width}, Match threshold: {self.match_threshold}")
        print(f"GPU usage: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"Interactive mode: {'Enabled' if self.interactive else 'Disabled'}")
        print(f"Ground truth database has {len(self.ground_truth_db)} entries")
    
    def _load_ground_truth(self):
        if os.path.exists(self.ground_truth_file):
            try:
                return pd.read_csv(self.ground_truth_file)
            except Exception as e:
                print(f"Error loading ground truth database: {e}")
                return pd.DataFrame(columns=['image_filename', 'ground_truth_name', 'ground_truth_floor', 'verified_date'])
        else:
            return pd.DataFrame(columns=['image_filename', 'ground_truth_name', 'ground_truth_floor', 'verified_date'])
    
    def _save_ground_truth(self):
        try:
            self.ground_truth_db.to_csv(self.ground_truth_file, index=False)
            print(f"Ground truth database saved with {len(self.ground_truth_db)} entries")
        except Exception as e:
            print(f"Error saving ground truth database: {e}")
    
    def _load_candidates_from_db(self):
        try:
            students = [(s.name, str(s.floor)) for s in Student.query.all()]
            staff = [(s.name, str(s.floor)) for s in Staff.query.all()]
            all_candidates = students + staff
            print(f"Loaded {len(students)} students and {len(staff)} staff members from database")
            return all_candidates
        except Exception as e:
            print(f"Error loading names from database: {e}")
            print("Falling back to dummy data")
            return [
                ("John Smith", "3"),
                ("Jane Doe", "5"),
                ("Robert Johnson", "2"),
                ("Emily Wilson", "4"),
                ("Michael Brown", "1"),
            ]
    
    def load_images(self):
        image_paths = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for filename in os.listdir(self.images_folder):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_paths.append(os.path.join(self.images_folder, filename))
        print(f"Found {len(image_paths)} images in {self.images_folder}")
        return sorted(image_paths)
    
    def run_test(self, pipeline, image_path, model_type):
        image_filename = os.path.basename(image_path)
        print(f"\nTesting {model_type} on {image_filename}")
        
        ground_truth_entry = self.ground_truth_db[self.ground_truth_db['image_filename'] == image_filename]
        if not ground_truth_entry.empty:
            ground_truth = (ground_truth_entry['ground_truth_name'].iloc[0],
                            ground_truth_entry['ground_truth_floor'].iloc[0])
            print(f"Found ground truth: {ground_truth[0]} (Floor: {ground_truth[1]})")
        else:
            ground_truth = None
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        try:
            text_path, annotated_image_path, recognized_lines = run_ocr_on_frame(
                pipeline=pipeline,
                frame=frame,
                results_folder=os.path.join(self.results_folder, model_type),
                target_width=self.ocr_preprocess_width
            )
            
            ocr_text_combined = "\n".join(recognized_lines)
            best_match, score = approximate_receiver_name(
                ocr_text_combined,
                self.candidate_names,
                noise_words,
                threshold=0.8,
                match_threshold=self.match_threshold
            )
            
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
        image_paths = self.load_images()
        for idx, image_path in enumerate(image_paths):
            print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            base_result = self.run_test(base_model_pipeline, image_path, "base_model")
            if base_result:
                self.results.append(base_result)
            trained_result = self.run_test(trained_model_pipeline, image_path, "trained_model")
            if trained_result:
                self.results.append(trained_result)
    
    def _save_visualization(self, image_filename, original_path):
        try:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            original_img = cv2.imread(original_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            ax.imshow(original_img)
            ax.set_title(f"Original: {image_filename}")
            ax.axis('off')
            vis_file = os.path.join(self.visualization_folder, f"{os.path.splitext(image_filename)[0]}_original.png")
            plt.tight_layout()
            plt.savefig(vis_file, dpi=150)
            plt.close(fig)
            return vis_file
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return None

    def compare_floors(self, predicted, ground_truth):
        """
        Compare two floor values numerically if possible.
        """
        try:
            return abs(float(predicted) - float(ground_truth)) < 1e-6
        except:
            return str(predicted).strip() == str(ground_truth).strip()
    
    def interactive_verification(self):
        df = pd.DataFrame(self.results)
        
        for idx, row in df.iterrows():
            image_filename = row['image_filename']
            gt_entry = self.ground_truth_db[self.ground_truth_db['image_filename'] == image_filename]
            if not gt_entry.empty:
                df.at[idx, 'ground_truth_name'] = gt_entry['ground_truth_name'].iloc[0]
                df.at[idx, 'ground_truth_floor'] = gt_entry['ground_truth_floor'].iloc[0]
                df.at[idx, 'has_ground_truth'] = True
        
        for image_filename, group in df.groupby('image_filename'):
            if self.interactive:
                os.system('cls' if os.name == 'nt' else 'clear')
            
            original_path = group['original_image_path'].iloc[0]
            try:
                if self.interactive:
                    plt.figure(figsize=(8, 6))
                    img = cv2.imread(original_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.title(f"Original Image: {image_filename}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show(block=False)
                else:
                    vis_path = self._save_visualization(image_filename, original_path)
                    if vis_path:
                        print(f"Saved visualization to: {vis_path}")
                
                if 'has_ground_truth' in group.columns and group['has_ground_truth'].any():
                    gt_row = group[group['has_ground_truth']].iloc[0]
                    ground_truth_name = gt_row['ground_truth_name']
                    ground_truth_floor = gt_row['ground_truth_floor']
                    print(f"\nExisting Ground Truth: {ground_truth_name} (Floor: {ground_truth_floor})")
                    for model_type in ['base_model', 'trained_model']:
                        model_result = group[group['model_type'] == model_type]
                        if not model_result.empty:
                            predicted_name = model_result['predicted_name'].iloc[0]
                            predicted_floor = model_result['predicted_floor'].iloc[0]
                            is_correct = (str(predicted_name).strip().lower() == str(ground_truth_name).strip().lower() and
                                          self.compare_floors(predicted_floor, ground_truth_floor))
                            idx = df.index[(df['image_filename'] == image_filename) & (df['model_type'] == model_type)].tolist()[0]
                            df.at[idx, 'is_correct'] = is_correct
                            print(f"{model_type} prediction ({predicted_name}, Floor: {predicted_floor}) auto-verified as {'correct' if is_correct else 'incorrect'}")
                    if self.interactive:
                        plt.close('all')
                        print("Auto-verification complete; continuing to next image...\n")
                else:
                    # For images without ground truth, first print out the predictions so you know what they are.
                    base_pred = group[group['model_type'] == 'base_model']
                    trained_pred = group[group['model_type'] == 'trained_model']
                    if not base_pred.empty:
                        print("Base model prediction: {} (Floor: {})".format(
                            base_pred['predicted_name'].iloc[0],
                            base_pred['predicted_floor'].iloc[0]
                        ))
                    if not trained_pred.empty:
                        print("Trained model prediction: {} (Floor: {})".format(
                            trained_pred['predicted_name'].iloc[0],
                            trained_pred['predicted_floor'].iloc[0]
                        ))
                    
                    print("\nNo ground truth available for this image.")
                    set_gt = input("Do you want to set ground truth? (y/n): ").lower().strip()
                    if set_gt == 'y':
                        print("\nSelect ground truth source:")
                        print("1. Use base model prediction")
                        print("2. Use trained model prediction")
                        print("3. Enter custom ground truth")
                        choice = input("Enter choice (1-3): ").strip()
                        if choice == '1':
                            base_row = group[group['model_type'] == 'base_model'].iloc[0]
                            ground_truth_name = base_row['predicted_name']
                            ground_truth_floor = base_row['predicted_floor']
                        elif choice == '2':
                            trained_row = group[group['model_type'] == 'trained_model'].iloc[0]
                            ground_truth_name = trained_row['predicted_name']
                            ground_truth_floor = trained_row['predicted_floor']
                        elif choice == '3':
                            ground_truth_name = input("Enter ground truth name: ").strip()
                            ground_truth_floor = input("Enter ground truth floor: ").strip()
                        else:
                            print("Invalid choice. Skipping ground truth setting.")
                            ground_truth_name = None
                            ground_truth_floor = None
                        
                        if ground_truth_name and ground_truth_name != "No match":
                            new_entry = pd.DataFrame({
                                'image_filename': [image_filename],
                                'ground_truth_name': [ground_truth_name],
                                'ground_truth_floor': [ground_truth_floor],
                                'verified_date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                            })
                            self.ground_truth_db = self.ground_truth_db[self.ground_truth_db['image_filename'] != image_filename]
                            self.ground_truth_db = pd.concat([self.ground_truth_db, new_entry], ignore_index=True)
                            self._save_ground_truth()
                            
                            print(f"Ground truth set: {ground_truth_name} (Floor: {ground_truth_floor})")
                            
                            for idx in df.index[df['image_filename'] == image_filename].tolist():
                                df.at[idx, 'ground_truth_name'] = ground_truth_name
                                df.at[idx, 'ground_truth_floor'] = ground_truth_floor
                                df.at[idx, 'has_ground_truth'] = True
                            
                            for model_type in ['base_model', 'trained_model']:
                                model_result = group[group['model_type'] == model_type]
                                if not model_result.empty:
                                    predicted_name = model_result['predicted_name'].iloc[0]
                                    predicted_floor = model_result['predicted_floor'].iloc[0]
                                    is_correct = (str(predicted_name).strip().lower() == str(ground_truth_name).strip().lower() and
                                                  self.compare_floors(predicted_floor, ground_truth_floor))
                                    idx = df.index[(df['image_filename'] == image_filename) & (df['model_type'] == model_type)].tolist()[0]
                                    df.at[idx, 'is_correct'] = is_correct
                                    print(f"{model_type} prediction ({predicted_name}, Floor: {predicted_floor}) auto-verified as {'correct' if is_correct else 'incorrect'}")
                        else:
                            print("Skipping ground truth setting for this image.")
                            for model_type in ['base_model', 'trained_model']:
                                idx = df.index[(df['image_filename'] == image_filename) & (df['model_type'] == model_type)].tolist()[0]
                                df.at[idx, 'is_correct'] = False
                    if self.interactive:
                        input("\nPress Enter to continue to the next image...")
            
            except Exception as e:
                print(f"Error during interactive verification for {image_filename}: {e}")
                if self.interactive:
                    input("Press Enter to continue...")
        
        return df
    
    def generate_report(self, verified_results):
        base_model_acc = verified_results[verified_results['model_type'] == 'base_model']['is_correct'].mean()
        trained_model_acc = verified_results[verified_results['model_type'] == 'trained_model']['is_correct'].mean()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'total_images': len(verified_results['image_filename'].unique()),
            'base_model_accuracy': round(base_model_acc * 100, 2),
            'trained_model_accuracy': round(trained_model_acc * 100, 2),
            'improvement': round((trained_model_acc - base_model_acc) * 100, 2),
            'detailed_results': verified_results,
            'ground_truth_count': len(self.ground_truth_db)
        }
        
        report_file = os.path.join(self.results_folder, f"accuracy_report_{timestamp}.csv")
        verified_results.to_csv(report_file, index=False)
        
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
        
        detailed_report_file = os.path.join(self.results_folder, f"detailed_report_{timestamp}.csv")
        report_df.to_csv(detailed_report_file, index=False)
        
        ground_truth_report_file = os.path.join(self.ground_truth_folder, f"ground_truth_{timestamp}.csv")
        self.ground_truth_db.to_csv(ground_truth_report_file, index=False)
        
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
    parser.add_argument('--width', type=int, default=640, help='OCR preprocess width')
    parser.add_argument('--threshold', type=float, default=0.36, help='Match threshold')
    parser.add_argument('--trained-model', default='models/recognizer_with_augment.h5', help='Path to trained recognizer model')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode (no plot display)')
    
    args = parser.parse_args()
    
    if args.non_interactive:
        matplotlib.use('Agg')
    
    tester = OCRAccuracyTester(
        images_folder=args.images,
        results_folder=args.results,
        use_gpu=args.gpu,
        target_width=args.width,
        match_threshold=args.threshold,
        interactive=not args.non_interactive
    )
    
    print("\nInitializing base model...")
    base_model_pipeline = init_ocr_pipeline(custom_recognizer_path=None, custom_detector_path=None)
    
    print("\nInitializing trained model...")
    trained_model_pipeline = init_ocr_pipeline(custom_recognizer_path=args.trained_model, custom_detector_path=None)
    
    print("\nStarting accuracy test...")
    tester.test_all_images(base_model_pipeline, trained_model_pipeline)
    
    print("\nStarting interactive verification...")
    verified_results = tester.interactive_verification()
    
    print("\nGenerating accuracy report...")
    report = tester.generate_report(verified_results)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
