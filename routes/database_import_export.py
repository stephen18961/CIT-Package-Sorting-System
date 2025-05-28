from flask import Blueprint, render_template
from flask import jsonify, make_response, render_template, redirect, url_for, request, session, flash
from models import *

import os
import csv
import shutil
from io import StringIO

db_import_export_bp = Blueprint('db_import_export_bp', __name__, template_folder='templates')

@db_import_export_bp.route('/import_export', methods=['GET', 'POST'])
def import_export():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        # Import actions
        if action == 'import_students':
            try:
                if 'file' not in request.files:
                    flash('No file selected', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                    
                file = request.files['file']
                if file.filename == '':
                    flash('No file selected', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                
                if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() == 'csv':
                    # Process the uploaded CSV file
                    stream = StringIO(file.stream.read().decode("UTF-8"))
                    reader = csv.reader(stream)
                    next(reader)  # Skip header row
                    new_count = 0
                    update_count = 0
                    for row in reader:
                        if len(row) >= 2:
                            name, floor = row[0], row[1]
                            # Check if student already exists
                            existing_student = Student.query.filter_by(name=name).first()
                            if existing_student:
                                # Update floor if student exists
                                existing_student.floor = floor
                                update_count += 1
                            else:
                                # Create new student if not exists
                                student = Student(name=name, floor=floor)
                                db.session.add(student)
                                new_count += 1
                    print(new_count, new_count)
                    db.session.commit()
                    flash(f'Import completed: {new_count} new students added, {update_count} existing students updated.', 'success')
                else:
                    flash('Invalid file format. Please upload a CSV file.', 'danger')
                
                return redirect(url_for('db_crud_bp.database_management'))
            except Exception as e:
                flash(f'Error importing students: {str(e)}', 'danger')
                return redirect(url_for('db_crud_bp.database_management'))
                
        elif action == 'import_staff':
            try:
                if 'file' not in request.files:
                    flash('No file selected', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                    
                file = request.files['file']
                if file.filename == '':
                    flash('No file selected', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                
                if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() == 'csv':
                    # Process the uploaded CSV file
                    stream = StringIO(file.stream.read().decode("UTF-8"))
                    reader = csv.reader(stream)
                    next(reader)  # Skip header row
                    new_count = 0
                    update_count = 0
                    for row in reader:
                        if len(row) >= 2:
                            name, floor = row[0], row[1]
                            # Check if staff already exists
                            existing_staff = Staff.query.filter_by(name=name).first()
                            if existing_staff:
                                # Update floor if staff exists
                                existing_staff.floor = floor
                                update_count += 1
                            else:
                                # Create new staff if not exists
                                staff = Staff(name=name, floor=floor)
                                db.session.add(staff)
                                new_count += 1
                    db.session.commit()
                    flash(f'Import completed: {new_count} new staff added, {update_count} existing staff updated.', 'success')
                else:
                    flash('Invalid file format. Please upload a CSV file.', 'danger')
                
                return redirect(url_for('db_crud_bp.database_management'))
            except Exception as e:
                flash(f'Error importing staff: {str(e)}', 'danger')
                return redirect(url_for('db_crud_bp.database_management'))
        
        # Export actions
        elif action == 'export_students':
            try:
                # ADDED: Debug logging
                print("Starting export_students process")
                
                students = Student.query.all()
                csv_data = StringIO()
                writer = csv.writer(csv_data)
                writer.writerow(['Name', 'Floor'])  # Header row
                for student in students:
                    writer.writerow([student.name, student.floor])
                
                # Create exports directory if it doesn't exist
                os.makedirs('exports', exist_ok=True)
                
                # Save to file
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'exports/students_export_{timestamp}.csv'
                
                # ADDED: Debug logging
                print(f"Writing to file: {filename}")
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    f.write(csv_data.getvalue())
                
                # MODIFIED: Using absolute path and additional debugging
                abs_path = os.path.abspath(filename)
                print(f"Sending file with absolute path: {abs_path}")
                
                if not os.path.exists(abs_path):
                    print(f"ERROR: File does not exist at {abs_path}")
                    flash(f'Error: Generated file not found', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                
                # Try simpler approach with direct file read
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        
                    response = make_response(file_content)
                    response.headers["Content-Disposition"] = f"attachment; filename=students_export_{timestamp}.csv"
                    response.headers["Content-type"] = "text/csv"
                    return response
                except Exception as e:
                    print(f"Error in alternative send method: {str(e)}")
                    flash(f'Error exporting students: {str(e)}', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
            except Exception as e:
                # ADDED: Debug logging 
                print(f"Exception in export_students: {str(e)}")
                flash(f'Error exporting students: {str(e)}', 'danger')
                return redirect(url_for('db_crud_bp.database_management'))
                
        elif action == 'export_staff':
            try:
                # ADDED: Debug logging
                print("Starting export_staff process")
                
                staff_list = Staff.query.all()
                csv_data = StringIO()
                writer = csv.writer(csv_data)
                writer.writerow(['Name', 'Floor'])  # Header row
                for staff in staff_list:
                    writer.writerow([staff.name, staff.floor])
                
                # Create exports directory if it doesn't exist
                os.makedirs('exports', exist_ok=True)
                
                # Save to file
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'exports/staff_export_{timestamp}.csv'
                
                # ADDED: Debug logging
                print(f"Writing to file: {filename}")
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    f.write(csv_data.getvalue())
                
                # MODIFIED: Using absolute path and additional debugging
                abs_path = os.path.abspath(filename)
                print(f"Sending file with absolute path: {abs_path}")
                
                if not os.path.exists(abs_path):
                    print(f"ERROR: File does not exist at {abs_path}")
                    flash(f'Error: Generated file not found', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                
                # Try simpler approach with direct file read
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        
                    response = make_response(file_content)
                    response.headers["Content-Disposition"] = f"attachment; filename=staff_export_{timestamp}.csv"
                    response.headers["Content-type"] = "text/csv"
                    return response
                except Exception as e:
                    print(f"Error in alternative send method: {str(e)}")
                    flash(f'Error exporting staff: {str(e)}', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
            except Exception as e:
                # ADDED: Debug logging
                print(f"Exception in export_staff: {str(e)}")
                flash(f'Error exporting staff: {str(e)}', 'danger')
                return redirect(url_for('db_crud_bp.database_management'))
                
        elif action == 'export_packages':
            try:
                # ADDED: Debug logging
                print("Starting export_packages process")
                
                packages = Package.query.all()
                csv_data = StringIO()
                writer = csv.writer(csv_data)
                writer.writerow(['ID', 'Receiver Name', 'Target Floor', 'Package Type', 'Notes', 'Timestamp', 'Status'])
                for package in packages:
                    writer.writerow([
                        package.id,
                        package.receiver_name,
                        package.target_floor,
                        package.package_type,
                        package.notes,
                        package.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        package.status
                    ])
                
                # Create exports directory if it doesn't exist
                os.makedirs('exports', exist_ok=True)
                
                # Save to file
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'exports/packages_export_{timestamp}.csv'
                
                # ADDED: Debug logging
                print(f"Writing to file: {filename}")
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    f.write(csv_data.getvalue())
                
                # MODIFIED: Using absolute path and additional debugging
                abs_path = os.path.abspath(filename)
                print(f"Sending file with absolute path: {abs_path}")
                
                if not os.path.exists(abs_path):
                    print(f"ERROR: File does not exist at {abs_path}")
                    flash(f'Error: Generated file not found', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                
                # Try simpler approach with direct file read
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        
                    response = make_response(file_content)
                    response.headers["Content-Disposition"] = f"attachment; filename=packages_export_{timestamp}.csv"
                    response.headers["Content-type"] = "text/csv"
                    return response
                except Exception as e:
                    print(f"Error in alternative send method: {str(e)}")
                    flash(f'Error exporting packages: {str(e)}', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
            except Exception as e:
                # ADDED: Debug logging
                print(f"Exception in export_packages: {str(e)}")
                flash(f'Error exporting packages: {str(e)}', 'danger')
                return redirect(url_for('db_crud_bp.database_management'))
        
        # Database backup action
        elif action == 'backup_database':
            try:
                # ADDED: Debug logging
                print("Starting backup_database process")
                
                # Create backup directory if it doesn't exist
                backup_dir = 'database_backups'
                os.makedirs(backup_dir, exist_ok=True)
                
                # MODIFIED: Fix hardcoded path issue
                db_path = 'instance/packages.db'  # Use correct path with forward slashes
                
                # ADDED: Debug logging
                print(f"Looking for database at: {db_path}")
                
                # Verify the database file exists
                if not os.path.exists(db_path):
                    print(f"ERROR: Database file not found at: {db_path}")
                    flash(f'Database file not found at: {db_path}', 'danger')
                    return redirect(url_for('db_crud_bp.database_management'))
                    
                # Create a timestamped backup file
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = os.path.join(backup_dir, f'packages_db_backup_{timestamp}.db')
                
                # ADDED: Debug logging
                print(f"Creating backup at: {backup_file}")
                
                # Copy the database file
                shutil.copy2(db_path, backup_file)
                
                # ADDED: Verify backup was created
                if os.path.exists(backup_file):
                    print(f"Backup successfully created at: {backup_file}")
                    flash(f'Database backup created successfully: {backup_file}', 'success')
                else:
                    print(f"ERROR: Backup file was not created at: {backup_file}")
                    flash(f'Error: Backup file was not created', 'danger')
                    
                return redirect(url_for('db_crud_bp.database_management'))
            except Exception as e:
                # ADDED: Debug logging
                print(f"Exception in backup_database: {str(e)}")
                flash(f'Error creating database backup: {str(e)}', 'danger')
                return redirect(url_for('db_crud_bp.database_management'))
    
    return redirect(url_for('db_crud_bp.database_management'))