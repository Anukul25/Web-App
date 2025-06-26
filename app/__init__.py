from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
import os

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    load_dotenv()
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:keshav@localhost:5432/bank_churn')
    print(f"Using DATABASE_URL: {database_url}")
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-default-secret-key')
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'connect_args': {'sslmode': 'require'}}  # Add for Neon

    db.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    from app.routes.main import bp as main_bp
    from app.routes.predict import predict_bp
    from app.routes.dashboard import dashboard_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp, url_prefix='/predict')
    app.register_blueprint(dashboard_bp)

    return app