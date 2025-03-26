from flask import Flask

# Create the Flask app instance
app = Flask(__name__)

# Load configuration (if any)
app.config["SECRET_KEY"] = "supersecretkey12345!@#$%"  # Replace with a real secret key

# Import routes after creating the app to avoid circular imports
from app import routes