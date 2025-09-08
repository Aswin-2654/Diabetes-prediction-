import os
import logging
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash
from ml_model import DiabetesPredictor

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "diabetes-prediction-secret-key")

# Initialize ML model
diabetes_predictor = DiabetesPredictor()

@app.route('/')
def index():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Extract form data
        pregnancies = float(request.form.get('pregnancies', 0))
        glucose = float(request.form.get('glucose', 0))
        blood_pressure = float(request.form.get('blood_pressure', 0))
        skin_thickness = float(request.form.get('skin_thickness', 0))
        insulin = float(request.form.get('insulin', 0))
        bmi = float(request.form.get('bmi', 0))
        diabetes_pedigree = float(request.form.get('diabetes_pedigree', 0))
        age = float(request.form.get('age', 0))
        
        # Validate inputs
        if glucose <= 0:
            flash('Glucose level must be greater than 0', 'error')
            return redirect(url_for('index'))
        
        if bmi <= 0:
            flash('BMI must be greater than 0', 'error')
            return redirect(url_for('index'))
        
        if age <= 0:
            flash('Age must be greater than 0', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        features = [pregnancies, glucose, blood_pressure, skin_thickness, 
                   insulin, bmi, diabetes_pedigree, age]
        
        prediction, probability = diabetes_predictor.predict(features)
        
        # Prepare result data
        result_data = {
            'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'input_data': {
                'pregnancies': pregnancies,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'skin_thickness': skin_thickness,
                'insulin': insulin,
                'bmi': bmi,
                'diabetes_pedigree': diabetes_pedigree,
                'age': age
            }
        }
        
        return render_template('result.html', result=result_data)
        
    except ValueError as e:
        flash('Please enter valid numeric values for all fields', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        flash('An error occurred during prediction. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/metrics')
def metrics():
    """Display model performance metrics"""
    try:
        # Get model metrics
        metrics_data = diabetes_predictor.get_metrics()
        
        # Generate confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics_data['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Not Diabetic', 'Diabetic'],
                   yticklabels=['Not Diabetic', 'Diabetic'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save plot to base64 string
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        confusion_matrix_img = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Generate feature importance plot
        feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                        'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
        
        plt.figure(figsize=(10, 6))
        importance_abs = [abs(coef) for coef in metrics_data['feature_importance']]
        colors = ['red' if coef < 0 else 'blue' for coef in metrics_data['feature_importance']]
        
        bars = plt.bar(feature_names, importance_abs, color=colors, alpha=0.7)
        plt.title('Feature Importance (Absolute Values)')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, importance in zip(bars, metrics_data['feature_importance']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot to base64 string
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        feature_importance_img = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return render_template('metrics.html', 
                             metrics=metrics_data,
                             confusion_matrix_img=confusion_matrix_img,
                             feature_importance_img=feature_importance_img)
        
    except Exception as e:
        logging.error(f"Metrics error: {str(e)}")
        flash('An error occurred while loading metrics. Please try again.', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    flash('An internal error occurred. Please try again.', 'error')
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
