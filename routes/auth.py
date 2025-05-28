from flask import Blueprint, render_template
from flask import jsonify, make_response, render_template, redirect, url_for, request, session, flash
from models import db, User

# Create a blueprint instance
auth_bp = Blueprint('auth_bp', __name__, template_folder='templates')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check the database for a matching user
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)

@auth_bp.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('auth_bp.login'))