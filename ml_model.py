import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_openml

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = 'diabetes_model.pkl'
        self.scaler_path = 'diabetes_scaler.pkl'
        self.X_test = None
        self.y_test = None
        
        # Load or train model
        self._load_or_train_model()
    
    def _load_data(self):
        """Load and preprocess the PIMA Indian Diabetes Dataset"""
        try:
            logging.info("Loading PIMA Indian Diabetes Dataset...")
            
            # Try to load from OpenML (dataset ID: 37)
            try:
                diabetes = fetch_openml(name='diabetes', version=1, as_frame=True, parser='auto')
                X = diabetes.data
                y = diabetes.target
                
                # Convert target to binary (0/1) if needed
                y = y.astype(int)
                
            except Exception as e:
                logging.warning(f"Could not load from OpenML: {e}")
                # Create synthetic dataset based on PIMA characteristics if OpenML fails
                logging.info("Creating representative dataset...")
                np.random.seed(42)
                n_samples = 768
                
                # Generate features based on PIMA dataset statistics
                pregnancies = np.random.poisson(3.8, n_samples)
                glucose = np.random.normal(120.9, 31.9, n_samples)
                blood_pressure = np.random.normal(69.1, 19.4, n_samples)
                skin_thickness = np.random.normal(20.5, 16.0, n_samples)
                insulin = np.random.normal(79.8, 115.2, n_samples)
                bmi = np.random.normal(32.0, 7.9, n_samples)
                diabetes_pedigree = np.random.gamma(2, 0.25, n_samples)
                age = np.random.gamma(2, 15, n_samples) + 21
                
                # Ensure positive values where appropriate
                glucose = np.clip(glucose, 44, 199)
                blood_pressure = np.clip(blood_pressure, 24, 122)
                skin_thickness = np.clip(skin_thickness, 7, 99)
                insulin = np.clip(insulin, 14, 846)
                bmi = np.clip(bmi, 18.2, 67.1)
                diabetes_pedigree = np.clip(diabetes_pedigree, 0.078, 2.42)
                age = np.clip(age, 21, 81)
                
                X = pd.DataFrame({
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': blood_pressure,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'DiabetesPedigreeFunction': diabetes_pedigree,
                    'Age': age
                })
                
                # Generate target based on risk factors
                risk_score = (
                    0.3 * (glucose > 126) +
                    0.2 * (bmi > 30) +
                    0.15 * (age > 45) +
                    0.1 * (pregnancies > 5) +
                    0.1 * (blood_pressure > 80) +
                    0.15 * diabetes_pedigree
                )
                
                # Add some randomness and convert to binary
                risk_score += np.random.normal(0, 0.1, n_samples)
                y = (risk_score > 0.4).astype(int)
            
            # Handle missing or zero values in critical features
            if isinstance(X, pd.DataFrame):
                # Replace zeros with median for certain features (common in PIMA dataset)
                zero_replacement_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
                for col in zero_replacement_cols:
                    if col in X.columns:
                        median_val = X[X[col] > 0][col].median()
                        X.loc[X[col] == 0, col] = median_val
            
            logging.info(f"Dataset loaded successfully. Shape: {X.shape}")
            logging.info(f"Target distribution: {np.bincount(y)}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise
    
    def _train_model(self):
        """Train the logistic regression model"""
        try:
            logging.info("Training new model...")
            
            # Load data
            X, y = self._load_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Store test data for metrics
            self.X_test = X_test
            self.y_test = y_test
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train logistic regression model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Calculate and log accuracy
            X_test_scaled = self.scaler.transform(X_test)
            accuracy = self.model.score(X_test_scaled, y_test)
            logging.info(f"Model trained successfully. Test accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
    
    def _load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                logging.info("Loading existing model...")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Load test data for metrics (retrain to get test data)
                X, y = self._load_data()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                self.X_test = X_test
                self.y_test = y_test
                
                logging.info("Model loaded successfully")
            else:
                self._train_model()
                
        except Exception as e:
            logging.error(f"Error loading/training model: {e}")
            # Fallback: train new model
            self._train_model()
    
    def predict(self, features):
        """Make prediction for given features"""
        try:
            # Convert to numpy array and reshape
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of diabetes
            
            return prediction, probability
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            raise
    
    def get_metrics(self):
        """Get model performance metrics"""
        try:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Test data not available")
            
            # Scale test data
            X_test_scaled = self.scaler.transform(self.X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Get feature importance (coefficients)
            feature_importance = self.model.coef_[0]
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'feature_importance': feature_importance,
                'test_size': len(self.y_test),
                'positive_cases': sum(self.y_test),
                'negative_cases': len(self.y_test) - sum(self.y_test)
            }
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise
