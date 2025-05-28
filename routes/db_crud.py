from flask import Blueprint, render_template
from flask import jsonify, make_response, render_template, redirect, url_for, request, session, flash
from models import *

import os
import time

app_start_time = time.time()

db_crud_bp = Blueprint('db_crud_bp', __name__, template_folder='templates')

@db_crud_bp.route('/database')
def database_management():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    # Get counts for the database management page
    students_count = Student.query.count()
    staff_count = Staff.query.count()
    packages_count = Package.query.count()
    pending_packages_count = Package.query.filter_by(status='Pending').count()
    delivered_packages_count = Package.query.filter_by(status='Delivered').count()
    ocr_packages_count = Package.query.filter_by(package_type='OCR').count()
    
    # Calculate the database size - fix path with proper separator
    db_path = os.path.join('instance', 'packages.db')
    db_size = "Unknown"
    if os.path.exists(db_path):
        size_bytes = os.path.getsize(db_path)
        if size_bytes < 1024:
            db_size = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            db_size = f"{size_bytes/1024:.2f} KB"
        else:
            db_size = f"{size_bytes/(1024*1024):.2f} MB"
    
    # Get the most recent backup from proper directory
    backup_dir = 'database_backups'  # Match the directory in the backup function
    last_backup = "Never"
    if os.path.exists(backup_dir):
        backup_files = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
                       if f.startswith('packages_db_backup_') and f.endswith('.db')]
        if backup_files:
            # Get the most recently created backup file
            most_recent = max(backup_files, key=os.path.getctime)
            backup_time = datetime.datetime.fromtimestamp(os.path.getctime(most_recent))
            last_backup = backup_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate uptime (using the global app_start_time)
    current_time = time.time()
    uptime_seconds = int(current_time - app_start_time)
    
    # Format uptime nicely
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        uptime = f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        uptime = f"{hours}h {minutes}m {seconds}s"
    else:
        uptime = f"{minutes}m {seconds}s"
    
    return render_template('database.html', 
                          students_count=students_count,
                          staff_count=staff_count,
                          packages_count=packages_count,
                          pending_packages_count=pending_packages_count,
                          delivered_packages_count=delivered_packages_count,
                          ocr_packages_count=ocr_packages_count,
                          db_size=db_size,
                          last_backup=last_backup,
                          uptime=uptime)

# Student CRUD routes
@db_crud_bp.route('/students')
def list_students():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    students = Student.query.order_by(Student.floor, Student.name).all()
    return render_template('students.html', students=students)

@db_crud_bp.route('/students/add', methods=['GET', 'POST'])
def add_student():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        floor = request.form.get('floor')
        
        if not name or not floor:
            flash('Name and floor are required', 'danger')
            return redirect(url_for('db_crud_bp.add_student'))
        
        new_student = Student(name=name, floor=floor)
        db.session.add(new_student)
        db.session.commit()
        flash('Student added successfully!', 'success')
        return redirect(url_for('db_crud_bp.list_students'))
    
    return render_template('student_form.html', student=None, action='Add')

@db_crud_bp.route('/students/edit/<int:id>', methods=['GET', 'POST'])
def edit_student(id):
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    student = Student.query.get_or_404(id)
    
    if request.method == 'POST':
        student.name = request.form.get('name')
        student.floor = request.form.get('floor')
        db.session.commit()
        flash('Student updated successfully!', 'success')
        return redirect(url_for('db_crud_bp.list_students'))
    
    return render_template('student_form.html', student=student, action='Edit')

@db_crud_bp.route('/students/delete/<int:id>', methods=['POST'])
def delete_student(id):
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    student = Student.query.get_or_404(id)
    db.session.delete(student)
    db.session.commit()
    flash('Student deleted successfully!', 'success')
    return redirect(url_for('db_crud_bp.list_students'))

# Staff CRUD routes
@db_crud_bp.route('/staff')
def list_staff():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    staff = Staff.query.order_by(Staff.floor, Staff.name).all()
    return render_template('staff.html', staff=staff)

@db_crud_bp.route('/staff/add', methods=['GET', 'POST'])
def add_staff():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        floor = request.form.get('floor')
        
        if not name or not floor:
            flash('Name and floor are required', 'danger')
            return redirect(url_for('db_crud_bp.add_staff'))
        
        new_staff = Staff(name=name, floor=floor)
        db.session.add(new_staff)
        db.session.commit()
        flash('Staff added successfully!', 'success')
        return redirect(url_for('db_crud_bp.list_staff'))
    
    return render_template('staff_form.html', staff=None, action='Add')

@db_crud_bp.route('/staff/edit/<int:id>', methods=['GET', 'POST'])
def edit_staff(id):
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    staff = Staff.query.get_or_404(id)
    
    if request.method == 'POST':
        staff.name = request.form.get('name')
        staff.floor = request.form.get('floor')
        db.session.commit()
        flash('Staff updated successfully!', 'success')
        return redirect(url_for('db_crud_bp.list_staff'))
    
    return render_template('staff_form.html', staff=staff, action='Edit')

@db_crud_bp.route('/staff/delete/<int:id>', methods=['POST'])
def delete_staff(id):
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    staff = Staff.query.get_or_404(id)
    db.session.delete(staff)
    db.session.commit()
    flash('Staff deleted successfully!', 'success')
    return redirect(url_for('db_crud_bp.list_staff'))

@db_crud_bp.route('/manual_input', methods=['GET', 'POST'])
def manual_input():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    
    success_message = None
    error_message = None
    
    if request.method == 'POST':
        try:
            receiver_name = request.form.get('receiver_name')
            target_floor = request.form.get('target_floor')
            package_type = request.form.get('package_type')
            notes = request.form.get('notes', '')
            
            if not receiver_name or not target_floor or not package_type:
                error_message = "Please fill all required fields."
            else:
                new_package = Package(
                    receiver_name=receiver_name,
                    target_floor=target_floor,
                    package_type=package_type,
                    notes=notes,
                    timestamp=datetime.datetime.now(),
                    status='Pending'
                )
                db.session.add(new_package)
                db.session.commit()

                success_message = "Package information has been successfully submitted."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
    
    # Get all people from both Students and Staff tables for auto-suggestions
    students = Student.query.all()
    staff = Staff.query.all()
    
    # Combine both lists into a single list for the template
    all_people = []
    for student in students:
        all_people.append({'name': student.name, 'floor': student.floor})
    for staff_member in staff:
        all_people.append({'name': staff_member.name, 'floor': staff_member.floor})
    

    return render_template('manual_input.html', 
                            all_people=all_people,
                            success_message=success_message,
                            error_message=error_message,
                            data={'target_floor': None})

@db_crud_bp.route('/view_packages')
def view_packages():
    if 'username' not in session:
        return redirect(url_for('auth_bp.login'))
    # Retrieve all package records from the database
    packages = Package.query.all()
    return render_template('packages.html', packages=packages)

@db_crud_bp.route('/edit_package/<int:package_id>', methods=['GET', 'POST'])
def edit_package(package_id):
    package = Package.query.get_or_404(package_id)
    
    if request.method == 'POST':
        # Update package information
        package.receiver_name = request.form['receiver_name']
        package.target_floor = request.form['target_floor']
        package.package_type = request.form['package_type']
        package.notes = request.form['notes']
        
        db.session.commit()
        flash('Package updated successfully!', 'success')
        return redirect(url_for('db_crud_bp.view_packages'))
    
    # Get all people from both Students and Staff tables for auto-suggestions
    students = Student.query.all()
    staff = Staff.query.all()
    
    # Combine both lists into a single list for the template
    all_people = []
    for student in students:
        all_people.append({'name': student.name, 'floor': student.floor})
    for staff_member in staff:
        all_people.append({'name': staff_member.name, 'floor': staff_member.floor})

    return render_template('edit_package.html',
                            package=package, 
                            all_people=all_people)

@db_crud_bp.route('/package/<int:package_id>/deliver', methods=['POST'])
def deliver_package(package_id):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        package = Package.query.get_or_404(package_id)
        package.status = 'Delivered'
        package.status_updated_at = datetime.datetime.now()  
        db.session.commit()
        return jsonify({
            'success': True,
            'status_updated_at': package.status_updated_at.strftime('%Y-%m-%d %H:%M:%S')  
        })
    return jsonify({'success': False, 'message': 'Invalid request'})

@db_crud_bp.route('/package/<int:package_id>/delete', methods=['POST'])
def delete_package(package_id):
    package = Package.query.get_or_404(package_id)
    db.session.delete(package)
    db.session.commit()
    flash('Package deleted successfully!', 'success')
    return redirect(url_for('db_crud_bp.view_packages'))