from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import datetime
from extensions import db

# Define the Package model
class Package(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    receiver_name = db.Column(db.String(80), nullable=False)
    target_floor = db.Column(db.String(10), nullable=False)
    package_type = db.Column(db.String(50), nullable=False)  # e.g., 'OCR' or 'Manual'
    notes = db.Column(db.Text, default='')
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    status = db.Column(db.String(20), default='Pending')
    status_updated_at = db.Column(db.DateTime, default=datetime.datetime.now)

# Define a simple User model for the admin account
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # In production, store a password hash

# Define Student model
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    floor = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)

# Define Staff model
class Staff(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    floor = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)

# settings
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ocr_preprocess_width = db.Column(db.Integer, default=640)
    match_threshold = db.Column(db.Float, default=0.36)
    use_gpu = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<Settings width={self.ocr_preprocess_width}, threshold={self.match_threshold}, gpu={self.use_gpu}>'
