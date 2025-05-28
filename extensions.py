from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'something unique and secret'  # Set this immediately after creating the app

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///packages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)