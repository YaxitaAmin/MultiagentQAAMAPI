"""
Predictive Performance Analytics Engine for AMAPI System
ML-powered predictions and bottleneck detection
"""

import time
import json
import uuid
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from loguru import logger


@dataclass
class PerformancePrediction:
    """Performance prediction result"""
    prediction_id: str
    prediction_type: str
    predicted_value: float
    confidence: float
    features_used: List[str]
    model_version: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class BottleneckDetection:
    """Bottleneck detection result"""
    bottleneck_id: str
    bottleneck_type: str
    severity: str  # low, medium, high, critical
    location: str
    impact_estimate: float
    suggested_resolution: str
    detection_confidence: float
    timestamp: float


@dataclass
class OptimizationSuggestion:
    """AI-generated optimization suggestion"""
    suggestion_id: str
    category: str
    title: str
    description: str
    expected_impact: float
    implementation_difficulty: str
    priority_score: float
    affected_components: List[str]
    estimated_time_to_implement: float


class PredictivePerformanceEngine:
    """
    ML-powered predictive analytics engine
    Predicts performance, detects bottlenecks, and suggests optimizations
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # ML Models
        self.success_predictor = None
        self.attention_forecaster = None
        self.bottleneck_detector = None
        self.performance_regressor = None
        
        # Data preprocessing
        self.feature_scaler = StandardScaler()
        self.is_models_trained = False
        
        # Prediction storage
        self.predictions: List[PerformancePrediction] = []
        self.bottlenecks: List[BottleneckDetection] = []
        self.optimization_suggestions: List[OptimizationSuggestion] = []
        
        # Training data
        self.training_data: List[Dict[str, Any]] = []
        self.feature_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.engine_metrics = {
            'predictions_made': 0,
            'prediction_accuracy': 0.0,
            'bottlenecks_detected': 0,
            'optimizations_suggested': 0,
            'model_performance': {},
            'feature_importance': {},
            'prediction_confidence_avg': 0.0
        }
        
        # Initialize feature definitions
        self._initialize_feature_definitions()
        
        logger.info("Predictive Performance Engine initialized")

    def _initialize_feature_definitions(self):
        """Initialize feature extraction definitions"""
        self.feature_definitions = {
            # Task characteristics
            'task_complexity': lambda data: data.get('task_complexity', 0.5),
            'task_type_encoded': lambda data: self._encode_task_type(data.get('task_type', 'unknown')),
            'expected_duration': lambda data: data.get('expected_duration', 30.0),
            'instruction_length': lambda data: len(data.get('instruction', '')),
            
            # Agent characteristics  
            'agent_experience': lambda data: data.get('agent_experience', 0),
            'agent_success_rate': lambda data: data.get('agent_success_rate', 0.5),
            'agent_avg_execution_time': lambda data: data.get('agent_avg_execution_time', 30.0),
            'agent_specialization_score': lambda data: data.get('agent_specialization_score', 0.5),
            
            # System characteristics
            'system_load': lambda data: data.get('system_load', 0.5),
            'memory_usage': lambda data: data.get('memory_usage', 0.5),
            'concurrent_tasks': lambda data: data.get('concurrent_tasks', 1),
            'device_performance_score': lambda data: data.get('device_performance_score', 0.8),
            
            # Historical features
            'recent_success_rate': lambda data: data.get('recent_success_rate', 0.5),
            'recent_avg_duration': lambda data: data.get('recent_avg_duration', 30.0),
            'error_rate_trend': lambda data: data.get('error_rate_trend', 0.0),
            'attention_efficiency_trend': lambda data: data.get('attention_efficiency_trend', 0.5),
            
            # Context features
            'time_of_day': lambda data: (time.time() % 86400) / 86400,  # Normalized hour
            'day_of_week': lambda data: (time.time() // 86400) % 7,
            'workflow_position': lambda data: data.get('workflow_position', 0),
            'retry_attempt': lambda data: data.get('retry_attempt', 0)
        }

    def _encode_task_type(self, task_type: str) -> float:
        """Encode task type as numerical value"""
        task_encoding = {
            'wifi_test': 0.1,
            'settings_navigation': 0.2,
            'app_launch': 0.3,
            'ui_interaction': 0.4,
            'verification': 0.5,
            'complex_workflow': 0.6,
            'custom_task': 0.7,
            'unknown': 0.0
        }
        return task_encoding.get(task_type, 0.0)

    def extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from input data"""
        try:
            features = []
            for feature_name, extractor in self.feature_definitions.items():
                try:
                    feature_value = extractor(data)
                    features.append(float(feature_value))
                except Exception as e:
                    logger.debug(f"Error extracting feature {feature_name}: {e}")
                    features.append(0.0)  # Default value
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros((1, len(self.feature_definitions)))

    async def train_models(self, training_data: List[Dict[str, Any]]) -> None:
        """Train all ML models with provided data"""
        try:
            if len(training_data) < 10:
                logger.warning("Insufficient training data for ML models")
                return
            
            logger.info(f"Training ML models with {len(training_data)} samples...")
            
            # Prepare training features and targets
            X_features = []
            y_success = []
            y_attention = []
            y_duration = []
            
            for sample in training_data:
                features = self.extract_features(sample).flatten()
                X_features.append(features)
                
                y_success.append(1 if sample.get('success', False) else 0)
                y_attention.append(sample.get('attention_used', 2.0))
                y_duration.append(sample.get('execution_time', 30.0))
            
            X = np.array(X_features)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train success predictor (classification)
            self.success_predictor = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.success_predictor.fit(X_scaled, y_success)
            
            # Train attention forecaster (regression)
            self.attention_forecaster = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.attention_forecaster.fit(X_scaled, y_attention)
            
            # Train performance regressor (duration prediction)
            self.performance_regressor = LinearRegression()
            self.performance_regressor.fit(X_scaled, y_duration)
            
            # Train bottleneck detector (anomaly detection style)
            self.bottleneck_detector = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.15,
                max_depth=4,
                random_state=42
            )
            
            # Create bottleneck labels (high duration or low success as bottlenecks)
            bottleneck_labels = [
                1 if (duration > np.percentile(y_duration, 75) or success == 0) else 0
                for duration, success in zip(y_duration, y_success)
            ]
            self.bottleneck_detector.fit(X_scaled, bottleneck_labels)
            
            self.is_models_trained = True
            self.training_data.extend(training_data)
            
            # Calculate model performance metrics
            await self._evaluate_model_performance(X_scaled, y_success, y_attention, y_duration)
            
            logger.info("✅ ML models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")

    async def _evaluate_model_performance(self, X: np.ndarray, y_success: List[int],
                                        y_attention: List[float], y_duration: List[float]) -> None:
        """Evaluate trained model performance"""
        try:
            # Success prediction accuracy
            success_predictions = self.success_predictor.predict(X)
            success_accuracy = np.mean(success_predictions == y_success)
            
            # Attention forecasting error
            attention_predictions = self.attention_forecaster.predict(X)
            attention_mae = np.mean(np.abs(attention_predictions - y_attention))
            
            # Duration prediction error
            duration_predictions = self.performance_regressor.predict(X)
            duration_mae = np.mean(np.abs(duration_predictions - y_duration))
            
            # Store performance metrics
            self.engine_metrics['model_performance'] = {
                'success_accuracy': success_accuracy,
                'attention_mae': attention_mae,
                'duration_mae': duration_mae,
                'model_training_samples': len(y_success)
            }
            
            # Feature importance (if available)
            if hasattr(self.success_predictor, 'feature_importances_'):
                feature_names = list(self.feature_definitions.keys())
                importance_dict = {
                    name: importance 
                    for name, importance in zip(feature_names, self.success_predictor.feature_importances_)
                }
                self.engine_metrics['feature_importance'] = importance_dict
            
            logger.info(f"Model performance: Success accuracy: {success_accuracy:.3f}, "
                       f"Attention MAE: {attention_mae:.2f}, Duration MAE: {duration_mae:.2f}")
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")

    async def predict_task_success(self, task_data: Dict[str, Any]) -> PerformancePrediction:
        """Predict task success probability"""
        try:
            if not self.is_models_trained:
                # Return default prediction
                return PerformancePrediction(
                    prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                    prediction_type='task_success',
                    predicted_value=0.5,
                    confidence=0.1,
                    features_used=[],
                    model_version='untrained',
                    timestamp=time.time(),
                    metadata={'note': 'Model not trained yet'}
                )
            
            # Extract and scale features
            features = self.extract_features(task_data)
            features_scaled = self.feature_scaler.transform(features)
            
            # Predict success probability
            success_proba = self.success_predictor.predict_proba(features_scaled)[0]
            success_probability = success_proba[1] if len(success_proba) > 1 else success_proba[0]
            
            # Calculate prediction confidence based on model uncertainty
            confidence = self._calculate_prediction_confidence(features_scaled, 'success')
            
            prediction = PerformancePrediction(
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                prediction_type='task_success',
                predicted_value=success_probability,
                confidence=confidence,
                features_used=list(self.feature_definitions.keys()),
                model_version='v1.0',
                timestamp=time.time(),
                metadata={
                    'task_type': task_data.get('task_type', 'unknown'),
                    'complexity': task_data.get('task_complexity', 0.5)
                }
            )
            
            self.predictions.append(prediction)
            self.engine_metrics['predictions_made'] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting task success: {e}")
            return PerformancePrediction(
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                prediction_type='task_success',
                predicted_value=0.5,
                confidence=0.1,
                features_used=[],
                model_version='error',
                timestamp=time.time(),
                metadata={'error': str(e)}
            )

    async def forecast_attention_demand(self, task_data: Dict[str, Any]) -> PerformancePrediction:
        """Forecast attention demand for upcoming task"""
        try:
            if not self.is_models_trained:
                return PerformancePrediction(
                    prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                    prediction_type='attention_demand',
                    predicted_value=2.0,  # Default attention
                    confidence=0.1,
                    features_used=[],
                    model_version='untrained',
                    timestamp=time.time(),
                    metadata={'note': 'Model not trained yet'}
                )
            
            features = self.extract_features(task_data)
            features_scaled = self.feature_scaler.transform(features)
            
            # Predict attention demand
            attention_demand = self.attention_forecaster.predict(features_scaled)[0]
            confidence = self._calculate_prediction_confidence(features_scaled, 'attention')
            
            prediction = PerformancePrediction(
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                prediction_type='attention_demand',
                predicted_value=max(0.1, attention_demand),  # Ensure positive
                confidence=confidence,
                features_used=list(self.feature_definitions.keys()),
                model_version='v1.0',
                timestamp=time.time(),
                metadata={
                    'task_complexity': task_data.get('task_complexity', 0.5),
                    'expected_duration': task_data.get('expected_duration', 30.0)
                }
            )
            
            self.predictions.append(prediction)
            self.engine_metrics['predictions_made'] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error forecasting attention demand: {e}")
            return PerformancePrediction(
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                prediction_type='attention_demand',
                predicted_value=2.0,
                confidence=0.1,
                features_used=[],
                model_version='error',
                timestamp=time.time(),
                metadata={'error': str(e)}
            )

    async def detect_bottlenecks(self, system_data: Dict[str, Any]) -> List[BottleneckDetection]:
        """Detect performance bottlenecks in the system"""
        try:
            bottlenecks = []
            
            if not self.is_models_trained:
                return bottlenecks
            
            # Extract features
            features = self.extract_features(system_data)
            features_scaled = self.feature_scaler.transform(features)
            
            # Predict bottleneck probability
            bottleneck_proba = self.bottleneck_detector.predict_proba(features_scaled)[0]
            bottleneck_probability = bottleneck_proba[1] if len(bottleneck_proba) > 1 else bottleneck_proba[0]
            
            if bottleneck_probability > 0.6:  # Bottleneck threshold
                # Identify bottleneck characteristics
                bottleneck_type, location = self._identify_bottleneck_source(system_data, features_scaled)
                severity = self._assess_bottleneck_severity(bottleneck_probability, system_data)
                
                bottleneck = BottleneckDetection(
                    bottleneck_id=f"bottleneck_{uuid.uuid4().hex[:8]}",
                    bottleneck_type=bottleneck_type,
                    severity=severity,
                    location=location,
                    impact_estimate=bottleneck_probability,
                    suggested_resolution=self._generate_bottleneck_resolution(bottleneck_type, system_data),
                    detection_confidence=bottleneck_probability,
                    timestamp=time.time()
                )
                
                bottlenecks.append(bottleneck)
                self.bottlenecks.append(bottleneck)
                self.engine_metrics['bottlenecks_detected'] += 1
            
            # Additional rule-based bottleneck detection
            rule_based_bottlenecks = await self._detect_rule_based_bottlenecks(system_data)
            bottlenecks.extend(rule_based_bottlenecks)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error detecting bottlenecks: {e}")
            return []

    def _identify_bottleneck_source(self, system_data: Dict[str, Any], 
                                  features: np.ndarray) -> Tuple[str, str]:
        """Identify the source of bottleneck"""
        # Use feature importance to identify likely bottleneck source
        if 'feature_importance' in self.engine_metrics:
            importance = self.engine_metrics['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Map features to bottleneck types
            feature_to_bottleneck = {
                'system_load': ('resource_contention', 'system_resources'),
                'memory_usage': ('memory_bottleneck', 'memory_subsystem'),
                'concurrent_tasks': ('concurrency_bottleneck', 'task_scheduler'),
                'agent_avg_execution_time': ('agent_performance', 'agent_execution'),
                'task_complexity': ('complexity_bottleneck', 'task_planning'),
                'device_performance_score': ('hardware_limitation', 'device_hardware')
            }
            
            for feature_name, _ in top_features:
                if feature_name in feature_to_bottleneck:
                    return feature_to_bottleneck[feature_name]
        
        # Default classification
        return 'performance_degradation', 'system_general'

    def _assess_bottleneck_severity(self, probability: float, system_data: Dict[str, Any]) -> str:
        """Assess bottleneck severity"""
        if probability > 0.9:
            return 'critical'
        elif probability > 0.8:
            return 'high'
        elif probability > 0.7:
            return 'medium'
        else:
            return 'low'

    def _generate_bottleneck_resolution(self, bottleneck_type: str, 
                                      system_data: Dict[str, Any]) -> str:
        """Generate resolution suggestion for bottleneck"""
        resolutions = {
            'resource_contention': 'Reduce concurrent task load or increase system resources',
            'memory_bottleneck': 'Optimize memory usage or increase available memory',
            'concurrency_bottleneck': 'Adjust task scheduling or reduce parallel execution',
            'agent_performance': 'Retrain agent or adjust task complexity',
            'complexity_bottleneck': 'Simplify tasks or improve planning algorithms',
            'hardware_limitation': 'Upgrade hardware or optimize for current device',
            'performance_degradation': 'Review system configuration and optimize bottlenecks'
        }
        
        return resolutions.get(bottleneck_type, 'General performance optimization needed')

    async def _detect_rule_based_bottlenecks(self, system_data: Dict[str, Any]) -> List[BottleneckDetection]:
        """Detect bottlenecks using rule-based logic"""
        bottlenecks = []
        
        try:
            # High system load bottleneck
            system_load = system_data.get('system_load', 0.5)
            if system_load > 0.85:
                bottlenecks.append(BottleneckDetection(
                    bottleneck_id=f"bottleneck_{uuid.uuid4().hex[:8]}",
                    bottleneck_type='high_system_load',
                    severity='high' if system_load > 0.95 else 'medium',
                    location='system_resources',
                    impact_estimate=system_load,
                    suggested_resolution='Reduce system load or scale resources',
                    detection_confidence=0.9,
                    timestamp=time.time()
                ))
            
            # Memory usage bottleneck
            memory_usage = system_data.get('memory_usage', 0.5)
            if memory_usage > 0.9:
                bottlenecks.append(BottleneckDetection(
                    bottleneck_id=f"bottleneck_{uuid.uuid4().hex[:8]}",
                    bottleneck_type='memory_exhaustion',
                    severity='critical' if memory_usage > 0.95 else 'high',
                    location='memory_subsystem',
                    impact_estimate=memory_usage,
                    suggested_resolution='Free memory or optimize memory usage',
                    detection_confidence=0.95,
                    timestamp=time.time()
                ))
            
            # Excessive concurrent tasks
            concurrent_tasks = system_data.get('concurrent_tasks', 1)
            if concurrent_tasks > 10:
                bottlenecks.append(BottleneckDetection(
                    bottleneck_id=f"bottleneck_{uuid.uuid4().hex[:8]}",
                    bottleneck_type='excessive_concurrency',
                    severity='medium',
                    location='task_scheduler',
                    impact_estimate=min(1.0, concurrent_tasks / 20.0),
                    suggested_resolution='Reduce concurrent task limit',
                    detection_confidence=0.8,
                    timestamp=time.time()
                ))
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error in rule-based bottleneck detection: {e}")
            return []

    async def generate_optimization_suggestions(self, system_data: Dict[str, Any],
                                              performance_history: List[Dict[str, Any]]) -> List[OptimizationSuggestion]:
        """Generate AI-powered optimization suggestions"""
        try:
            suggestions = []
            
            # Analyze performance trends
            if len(performance_history) >= 5:
                recent_performance = performance_history[-10:]
                
                # Success rate trend analysis
                success_rates = [p.get('success_rate', 0.5) for p in recent_performance]
                success_trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
                
                if success_trend < -0.05:  # Declining success rate
                    suggestions.append(OptimizationSuggestion(
                        suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                        category='performance_improvement',
                        title='Address Declining Success Rate',
                        description='Success rate has been declining. Consider retraining agents or adjusting task complexity.',
                        expected_impact=0.8,
                        implementation_difficulty='medium',
                        priority_score=0.9,
                        affected_components=['agents', 'task_planning'],
                        estimated_time_to_implement=2.0
                    ))
                
                # Execution time trend analysis
                exec_times = [p.get('avg_execution_time', 30.0) for p in recent_performance]
                time_trend = np.polyfit(range(len(exec_times)), exec_times, 1)[0]
                
                if time_trend > 2.0:  # Increasing execution time
                    suggestions.append(OptimizationSuggestion(
                        suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                        category='efficiency_optimization',
                        title='Optimize Execution Speed',
                        description='Execution times are increasing. Consider optimizing algorithms or system resources.',
                        expected_impact=0.7,
                        implementation_difficulty='high',
                        priority_score=0.7,
                        affected_components=['execution_engine', 'system_resources'],
                        estimated_time_to_implement=4.0
                    ))
            
            # System resource optimization
            system_load = system_data.get('system_load', 0.5)
            if system_load > 0.7:
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                    category='resource_optimization',
                    title='Optimize Resource Usage',
                    description='System load is high. Consider load balancing or resource scaling.',
                    expected_impact=0.6,
                    implementation_difficulty='medium',
                    priority_score=0.8,
                    affected_components=['system_resources', 'task_scheduler'],
                    estimated_time_to_implement=1.5
                ))
            
            # Attention efficiency optimization
            if 'attention_efficiency_trend' in system_data:
                efficiency_trend = system_data['attention_efficiency_trend']
                if efficiency_trend < 0.6:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_id=f"opt_{uuid.uuid4().hex[:8]}",
                        category='attention_optimization',
                        title='Improve Attention Allocation',
                        description='Attention efficiency is below optimal. Consider rebalancing attention strategies.',
                        expected_impact=0.5,
                        implementation_difficulty='low',
                        priority_score=0.6,
                        affected_components=['attention_system', 'agents'],
                        estimated_time_to_implement=1.0
                    ))
            
            # Store suggestions
            self.optimization_suggestions.extend(suggestions)
            self.engine_metrics['optimizations_suggested'] += len(suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {e}")
            return []

    def _calculate_prediction_confidence(self, features: np.ndarray, prediction_type: str) -> float:
        """Calculate confidence in prediction based on model uncertainty"""
        try:
            # Use prediction variance as uncertainty measure
            if prediction_type == 'success' and hasattr(self.success_predictor, 'predict_proba'):
                probas = self.success_predictor.predict_proba(features)[0]
                # Higher variance in probabilities means lower confidence
                confidence = 1.0 - np.var(probas)
            elif prediction_type == 'attention' and hasattr(self.attention_forecaster, 'estimators_'):
                # For random forest, use prediction variance across trees
                predictions = [tree.predict(features)[0] for tree in self.attention_forecaster.estimators_[:10]]
                variance = np.var(predictions)
                confidence = max(0.1, 1.0 - variance / 10.0)  # Normalize variance
            else:
                confidence = 0.7  # Default confidence
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.debug(f"Error calculating prediction confidence: {e}")
            return 0.5

    async def update_prediction_accuracy(self, prediction_id: str, actual_outcome: Dict[str, Any]) -> None:
        """Update prediction accuracy with actual outcome"""
        try:
            # Find the prediction
            prediction = None
            for pred in self.predictions:
                if pred.prediction_id == prediction_id:
                    prediction = pred
                    break
            
            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found for accuracy update")
                return
            
            # Calculate accuracy based on prediction type
            if prediction.prediction_type == 'task_success':
                actual_success = 1 if actual_outcome.get('success', False) else 0
                predicted_success = 1 if prediction.predicted_value > 0.5 else 0
                accuracy = 1.0 if actual_success == predicted_success else 0.0
                
            elif prediction.prediction_type == 'attention_demand':
                actual_attention = actual_outcome.get('attention_used', 2.0)
                predicted_attention = prediction.predicted_value
                error = abs(actual_attention - predicted_attention)
                accuracy = max(0.0, 1.0 - error / max(1.0, actual_attention))
                
            else:
                accuracy = 0.5  # Default for unknown types
            
            # Update running average accuracy
            current_accuracy = self.engine_metrics['prediction_accuracy']
            total_predictions = self.engine_metrics['predictions_made']
            
            if total_predictions > 0:
                self.engine_metrics['prediction_accuracy'] = (
                    (current_accuracy * (total_predictions - 1) + accuracy) / total_predictions
                )
            else:
                self.engine_metrics['prediction_accuracy'] = accuracy
            
            logger.debug(f"Updated prediction accuracy: {accuracy:.2f} (overall: {self.engine_metrics['prediction_accuracy']:.2f})")
            
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}")

    def get_engine_analytics(self) -> Dict[str, Any]:
        """Get comprehensive predictive engine analytics"""
        try:
            # Recent predictions analysis
            recent_predictions = [p for p in self.predictions 
                                if time.time() - p.timestamp < 3600]  # Last hour
            
            prediction_types = defaultdict(int)
            confidence_by_type = defaultdict(list)
            
            for pred in recent_predictions:
                prediction_types[pred.prediction_type] += 1
                confidence_by_type[pred.prediction_type].append(pred.confidence)
            
            # Average confidence by type
            avg_confidence_by_type = {
                pred_type: np.mean(confidences) 
                for pred_type, confidences in confidence_by_type.items()
            }
            
            # Recent bottlenecks analysis
            recent_bottlenecks = [b for b in self.bottlenecks 
                                if time.time() - b.timestamp < 3600]  # Last hour
            
            bottleneck_severity = defaultdict(int)
            for bottleneck in recent_bottlenecks:
                bottleneck_severity[bottleneck.severity] += 1
            
            return {
                'engine_metrics': self.engine_metrics.copy(),
                'model_status': {
                    'is_trained': self.is_models_trained,
                    'training_samples': len(self.training_data),
                    'features_count': len(self.feature_definitions)
                },
                'recent_activity': {
                    'predictions_last_hour': len(recent_predictions),
                    'prediction_types': dict(prediction_types),
                    'avg_confidence_by_type': avg_confidence_by_type,
                    'bottlenecks_last_hour': len(recent_bottlenecks),
                    'bottleneck_severity_distribution': dict(bottleneck_severity)
                },
                'optimization_suggestions': {
                    'total_suggestions': len(self.optimization_suggestions),
                    'categories': list(set(s.category for s in self.optimization_suggestions)),
                    'avg_expected_impact': np.mean([s.expected_impact for s in self.optimization_suggestions]) if self.optimization_suggestions else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating predictive engine analytics: {e}")
            return {'error': str(e)}


__all__ = [
    "PredictivePerformanceEngine",
    "PerformancePrediction",
    "BottleneckDetection", 
    "OptimizationSuggestion"
]