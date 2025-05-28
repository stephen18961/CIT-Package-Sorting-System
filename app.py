import time
from flask import jsonify, make_response, render_template, redirect, url_for, request, session, flash
import datetime
import base64
import cv2
import numpy as np
import os
import requests
from sqlalchemy import extract

# app and models
from extensions import app, db
from models import *

# Create the database tables if they don't exist yet
with app.app_context():
    db.create_all()
    # Create a default admin account if it doesn't exist
    if not User.query.filter_by(username='admin').first():
        admin_user = User(username='admin', password='password')
        db.session.add(admin_user)
        db.session.commit()
    
    # Create default settings if they don't exist
    settings = Settings.query.first()
    if not settings:
        # We'll determine GPU availability later
        default_settings = Settings(
            ocr_preprocess_width=640,
            match_threshold=0.36,
            use_gpu=False  # Default to CPU to be safe
        )
        db.session.add(default_settings)
        db.session.commit()
        settings = default_settings

# WeMos ESP8266 IP
IOT_DEVICE_IP = "http://192.168.1.17"

# Import OCR-related modules
from predictor import approximate_receiver_name, noise_words
from ocr_module import init_ocr_pipeline, run_ocr_on_frame

import tensorflow as tf
# Check if GPU is available (for informational purposes only)
gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0

RESULTS_FOLDER = "keras_ocr_results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize Keras-OCR pipeline
pipeline = init_ocr_pipeline(
    custom_recognizer_path='models/recognizer_with_augment.h5',
    custom_detector_path=None,
    use_gpu=settings.use_gpu
)

# Load candidate names
def get_candidates_from_db():
    students = [(s.name, str(s.floor)) for s in Student.query.all()]
    staff = [(s.name, str(s.floor)) for s in Staff.query.all()]
    return students + staff

@app.context_processor
def inject_admin():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        return {'admin_name': user.username if user else ''}
    return {'admin_name': 'Guest'}

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    return redirect(url_for('dashboard'))

from routes.auth import auth_bp
app.register_blueprint(auth_bp)
from routes.database_import_export import db_import_export_bp
app.register_blueprint(db_import_export_bp)
from routes.db_crud import db_crud_bp
app.register_blueprint(db_crud_bp)

def check_servo_status(IOT_DEVICE_IP):
    """
    Check the status of the servo by pinging the WeMos IP.
    
    Args:
        IOT_DEVICE_IP (str): The IP address of the WeMos ESP8266 device
    
    Returns:
        str: 'Active' if the device is reachable, 'Inactive' if not
    """
    try:
        # Use requests with a short timeout to quickly check device availability
        response = requests.get(f"{IOT_DEVICE_IP}/status", timeout=2)
        
        # Check if the response is successful
        if response.status_code == 200:
            return 'Active'
        else:
            return 'Inactive'
    
    except (requests.exceptions.RequestException, 
            requests.exceptions.Timeout, 
            requests.exceptions.ConnectionError):
        return 'Inactive'

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    servo_status = check_servo_status(IOT_DEVICE_IP)
    # Get admin info from the database
    admin_user = User.query.filter_by(username=session['username']).first()
    # Compute package statistics dynamically
    total_packages = Package.query.count()
    auto_sorted = Package.query.filter_by(package_type='OCR').count()
    manual_input = Package.query.filter(Package.package_type != 'OCR').count()
    last_ocr = Package.query.filter_by(package_type='OCR').order_by(Package.timestamp.desc()).first()
    
    # Build the dashboard data
    data = {
        'admin_name': admin_user.username if admin_user else '',
        'camera_status': 'Online',
        'servo_status': servo_status,
        'last_update': datetime.datetime.now().strftime('%H:%M:%S'),
        'total_packages': total_packages,
        'auto_sorted': auto_sorted,
        'manual_input': manual_input,
        'ocr_text': last_ocr.receiver_name if last_ocr else None,
        'target_floor': last_ocr.target_floor if last_ocr else None,
        'process_status': 'Ready for Scanning'
    }
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    return render_template('dashboard.html', data=data, current_time=current_time)

@app.route('/process_ocr', methods=['POST'])
def process_ocr():
    if 'username' not in session:
        return {'error': 'Unauthorized'}, 401

    data = request.get_json()
    if not data or 'image' not in data:
        return {'error': 'No image data received'}, 400

    try:
        header, encoded = data['image'].split(',', 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return _handle_ocr_failure(IOT_DEVICE_IP, 'Failed to decode image')

        print(f"\nRunning OCR...")
        
        # Retrieve settings from the database
        settings_record = Settings.query.first()
        ocr_width = settings_record.ocr_preprocess_width if settings_record else 640
        match_threshold = settings_record.match_threshold if settings_record else 0.36

        # Run OCR on the received frame using the target width from settings
        text_path, annotated_image_path, recognized_lines = run_ocr_on_frame(
            pipeline=pipeline,
            frame=frame,
            results_folder=RESULTS_FOLDER,
            target_width=ocr_width
        )
        ocr_text_combined = "\n".join(recognized_lines)

        print(f"\nRunning Approximation...")
        # Get updated candidates from database
        candidate_names = get_candidates_from_db()
        
        # Run the rule-based predictor on the OCR result, passing the match_threshold from settings
        best_match, score = approximate_receiver_name(
            ocr_text_combined,
            candidate_names,
            noise_words,
            threshold=0.8,  # This is the noise word threshold, keep it as is
            match_threshold=match_threshold  # Pass the match threshold from settings
        )

        if best_match:
            print(f"Best match: {best_match} (score: {score:.4f})")
        else:
            print("No matching name found.")
            return _handle_ocr_failure(IOT_DEVICE_IP, 'No matching name found')

        name_detected = best_match[0]
        floor_detected = best_match[1]
        # Save the OCR package data to SQLite
        new_package = Package(
            receiver_name=name_detected,
            target_floor=floor_detected,
            package_type='OCR',           # Mark this package as processed via OCR
            notes='Processed via OCR',
            timestamp=datetime.datetime.now(),
            status='Pending'
        )
        db.session.add(new_package)
        db.session.commit()

        # Check servo status before attempting to send data
        servo_status = check_servo_status(IOT_DEVICE_IP)
        
        response_data = {
            'status': 'Success',
            'receiver_name': name_detected,
            'target_floor': floor_detected,
            'confidence_score': score,
            'servo_status': servo_status,
            'threshold_used': match_threshold
        }

        if servo_status == 'Active':
            try:
                response = requests.post(f"{IOT_DEVICE_IP}/sort", json={"floor": floor_detected}, timeout=5)
                esp_response = response.json()
                print(f"ESP Response: {esp_response}")
                response_data['esp_response'] = esp_response
            except requests.exceptions.RequestException as e:
                response_data['esp_error'] = str(e)
                print(f"ESP Request Error: {e}")

        return response_data

    except Exception as e:
        return _handle_ocr_failure(IOT_DEVICE_IP, str(e))

def _handle_ocr_failure(iot_device_ip, error_message):
    """
    Handle OCR failure 
    
    Args:
        iot_device_ip (str): IP address of the IoT device
        error_message (str): Description of the error
    
    Returns:
        dict: Response with error details
    """
    print(f"OCR Failure: {error_message}")
    
    # Check servo status 
    servo_status = check_servo_status(iot_device_ip)
    
    response_data = {
        'status': 'Failed',
        'receiver_name': '',
        'target_floor': '',
        'error': error_message,
        'servo_status': servo_status
    }

    if servo_status == 'Active':
        try:
            # Send '999' to trigger error beeping
            response = requests.post(f"{iot_device_ip}/sort", json={"floor": "999"}, timeout=5)
            esp_response = response.json()
            print(f"Error Signal ESP Response: {esp_response}")
            response_data['esp_response'] = esp_response
        except requests.exceptions.RequestException as e:
            response_data['esp_error'] = str(e)
            print(f"Failed to send error signal to ESP8266: {e}")

    return response_data

@app.route('/get_stats_by_date', methods=['GET'])
def get_stats_by_date():
    """
    Get filtered statistics based on date range
    """
    try:
        date_range = request.args.get('range', 'all')
        
        # Calculate the date filter based on the range
        today = datetime.datetime.now().date()
        if date_range == 'week':
            start_date = today - datetime.timedelta(days=7)
        elif date_range == 'month':
            start_date = today - datetime.timedelta(days=30)
        else:  # 'all'
            start_date = None
        
        # Generate filtered statistics
        stats = generate_package_stats(start_date)
        
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error in get_stats_by_date: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_package_stats(start_date=None):
    """
    Generate comprehensive package statistics from database
    with optional date filtering
    """
    # Apply date filter to the base query if start_date is provided
    base_query = Package.query
    if start_date:
        base_query = base_query.filter(db.func.date(Package.timestamp) >= start_date)
    
    # Total Packages
    total_packages = base_query.count()
    delivered_packages = base_query.filter_by(status='Delivered').count()
    delivery_rate = round((delivered_packages / total_packages) * 100, 2) if total_packages > 0 else 0
    pending_packages = base_query.filter_by(status='Pending').count()

    # Temporal Statistics
    def get_packages_by_hour():
        hourly_packages = {}
        for hour in range(24):
            query = base_query.filter(extract('hour', Package.timestamp) == hour)
            count = query.count()
            hourly_packages[f"{hour:02d}:00"] = count
        return hourly_packages

    def get_packages_by_day():
        daily_packages = {}
        for days_ago in range(7):
            date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).date()
            query = base_query.filter(db.func.date(Package.timestamp) == date)
            count = query.count()
            daily_packages[date.strftime('%Y-%m-%d')] = count
        return daily_packages

    # Floor Distribution
    floor_distribution = {}
    all_floors = base_query.with_entities(Package.target_floor).distinct()
    for floor in all_floors:
        floor_query = base_query.filter(Package.target_floor == floor[0])
        floor_count = floor_query.count()
        floor_distribution[floor[0]] = round((floor_count / total_packages) * 100, 2) if total_packages > 0 else 0
        
    # Package Type Breakdown
    ocr_packages = base_query.filter_by(package_type='OCR').count()
    manual_packages = base_query.filter(Package.package_type != 'OCR').count()
    package_type_breakdown = {
        "ocr_vs_manual": {
            "OCR Processing": round((ocr_packages / total_packages) * 100, 2) if total_packages > 0 else 0,
            "Manual Processing": round((manual_packages / total_packages) * 100, 2) if total_packages > 0 else 0
        }
    }

    # Performance Metrics
    avg_processing_time = _calculate_average_processing_time(base_query)
    delivery_efficiency = round((delivered_packages / total_packages) * 100, 2) if total_packages > 0 else 0

    # Peak Periods with date-filtered query
    peak_periods = _get_peak_periods(base_query)

    return {
        "kpis": {
            "total_packages": total_packages,
            "delivered_packages": delivered_packages,
            "delivery_rate": delivery_rate,
            "pending_packages": pending_packages
        },
        "temporal_stats": {
            "packages_by_hour": get_packages_by_hour(),
            "packages_by_day": get_packages_by_day(),
            "peak_periods": peak_periods,
        },
        "floor_distribution": floor_distribution,
        "package_type_breakdown": package_type_breakdown,
        "time_metrics": {
            "avg_processing_time": round(avg_processing_time, 2)
        },
        "performance_metrics": {
            "delivery_efficiency": delivery_efficiency,
            "avg_handling_time": round(avg_processing_time / 60, 2),  # Convert to hours
        }
    }

def _calculate_average_processing_time(query=None):
    """
    Calculate average processing time in minutes with optional query filter
    """
    try:
        if query is None:
            query = Package.query
            
        processing_times = [
            (pkg.status_updated_at - pkg.timestamp).total_seconds() / 60 
            for pkg in query.all() 
            if pkg.status_updated_at and pkg.timestamp
        ]
        return sum(processing_times) / len(processing_times) if processing_times else 0
    except Exception:
        return 0

def _get_peak_periods(query=None):
    """
    Identify peak periods for package processing using SQLAlchemy's extract function
    with optional query filter
    """
    if query is None:
        query = Package.query
        
    peak_periods = {}
    for period, (start_hour, end_hour) in [
        ("Morning (6-9 AM)", (6, 9)),
        ("Afternoon (12-2 PM)", (12, 14)),
        ("Evening (5-7 PM)", (17, 19))
    ]:
        period_query = query.filter(
            extract('hour', Package.timestamp).between(start_hour, end_hour)
        )
        count = period_query.count()
        peak_periods[period] = {"count": count}
    return peak_periods

@app.route('/advanced_statistics')
def advanced_statistics():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    # Get date range from query parameters (if any)
    date_range = request.args.get('range', 'all')
    
    # Calculate the date filter based on the range
    start_date = None
    if date_range == 'week':
        start_date = datetime.datetime.now().date() - datetime.timedelta(days=7)
    elif date_range == 'month':
        start_date = datetime.datetime.now().date() - datetime.timedelta(days=30)
    
    # Generate statistics with the optional date filter
    stats = generate_package_stats(start_date)
    
    # Add date range info to the stats
    stats['current_range'] = date_range
    
    return render_template('advanced_statistics.html', stats=stats)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    # Check if GPU is available
    import tensorflow as tf
    gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    
    # Get the current settings
    settings = Settings.query.first()
    if not settings:
        # Create default settings if not exists
        settings = Settings(
            ocr_preprocess_width=640, 
            match_threshold=0.36,
            use_gpu=False  # Default to CPU to be safe
        )
        db.session.add(settings)
        db.session.commit()
    
    if request.method == 'POST':
        try:
            # Process OCR width settings
            new_width = int(request.form.get('ocr_preprocess_width', 640))
            
            # Process match threshold settings
            new_threshold = float(request.form.get('match_threshold', 0.36))
            
            # Process GPU usage setting
            new_use_gpu = request.form.get('use_gpu') == '1'
            
            # Only allow GPU if it's available
            if new_use_gpu and not gpu_available:
                new_use_gpu = False
                flash('GPU is not available on this system. Defaulting to CPU.', 'warning')
            
            # Validate and save settings
            if new_width >= 0:
                settings.ocr_preprocess_width = new_width
                
                if new_width == 0:
                    flash_message = 'Image preprocessing disabled. Original image will be used.'
                else:
                    flash_message = f'OCR preprocessing width updated to {new_width}px.'
            else:
                flash('Invalid width. Please enter 0 or a positive number.', 'error')
                return redirect(url_for('settings'))
            
            # Validate and save threshold
            if 0 <= new_threshold <= 1:
                settings.match_threshold = new_threshold
                flash_message += f' Match threshold updated to {new_threshold:.3f}.'
            else:
                flash('Invalid threshold. Please enter a value between 0 and 1.', 'error')
                return redirect(url_for('settings'))
            
            # Save GPU setting
            old_gpu_setting = settings.use_gpu
            settings.use_gpu = new_use_gpu
            
            # Modify preprocessing function to handle 0 or negative values
            global preprocess_image
            def preprocess_image(frame, target_width=new_width):
                """
                Resize image while preserving aspect ratio for optimal OCR processing.
                If target_width is 0 or negative, return the original image.
                """
                if target_width <= 0:
                    return frame  # Return original image without preprocessing
                
                # Get original dimensions
                h, w = frame.shape[:2]
                
                # Compute the new height to maintain the aspect ratio
                aspect_ratio = h / w
                new_height = int(target_width * aspect_ratio)
                
                # Resize while keeping aspect ratio
                resized_image = cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)
                
                return resized_image
            
            # If GPU setting changed, we need to notify the user to restart the application
            if old_gpu_setting != new_use_gpu:
                flash_message += ' GPU/CPU setting changed. Please restart the application for changes to take effect.'
            
            db.session.commit()
            flash(flash_message, 'success')
            
        except ValueError:
            flash('Invalid input. Please enter valid numbers.', 'error')
        
        return redirect(url_for('settings'))
    
    return render_template('settings.html', 
                          current_width=settings.ocr_preprocess_width,
                          current_threshold=settings.match_threshold,
                          current_use_gpu=settings.use_gpu,
                          gpu_available=gpu_available)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)