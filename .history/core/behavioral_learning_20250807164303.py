"""
Behavioral Pattern Engine for AMAPI System
Learns from success/failure patterns and predicts outcomes
"""

import time
import json
import uuid
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from loguru import logger


@dataclass
class BehavioralPattern:
    """Individual behavioral pattern"""
    pattern_id: str
    pattern_type: str
    success_rate: float
    failure_rate: float
    attention_cost: float
    confidence: float
    frequency: int
    last_seen: float
    context_factors: Dict[str, Any]
    learning_metadata: Dict[str, Any]


@dataclass
class AttentionSuccessCorrelation:
    """Correlation between attention usage and success"""
    attention_range: Tuple[float, float]  # (min, max)
    success_probability: float
    sample_size: int
    optimal_attention: float
    efficiency_score: float


class BehavioralPatternEngine:
    """
    Core behavioral pattern learning engine
    Maps attention allocation strategies to success outcomes
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Pattern storage
        self.behavioral_patterns: Dict[str, BehavioralPattern] = {}
        self.attention_correlations: Dict[str, AttentionSuccessCorrelation] = {}
        self.failure_predictors: Dict[str, Dict[str, Any]] = {}
        
        # Learning state
        self.pattern_frequency: Dict[str, int] = defaultdict(int)
        self.success_tracking: List[Dict[str, Any]] = []
        self.attention_history: List[Dict[str, Any]] = []
        
        # Adaptive verification criteria
        self.verification_criteria: Dict[str, Dict[str, Any]] = {}
        self.criteria_adaptation_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.engine_metrics = {
            'patterns_discovered': 0,
            'predictions_made': 0,
            'prediction_accuracy': 0.0,
            'attention_optimization_impact': 0.0,
            'verification_adaptations': 0,
            'pattern_recognition_accuracy': 0.0
        }
        
        logger.info("Behavioral Pattern Engine initialized")

    async def learn_from_execution(self, agent_id: str, task_data: Dict[str, Any],
                                  execution_result: Dict[str, Any],
                                  attention_data: Dict[str, Any]) -> List[BehavioralPattern]:
        """Learn behavioral patterns from task execution"""
        discovered_patterns = []
        
        try:
            # Extract pattern features
            pattern_features = self._extract_pattern_features(task_data, execution_result, attention_data)
            
            # Identify pattern type
            pattern_type = self._classify_pattern_type(pattern_features)
            
            # Check for existing similar patterns
            existing_pattern = self._find_similar_pattern(pattern_features, pattern_type)
            
            if existing_pattern:
                # Update existing pattern
                updated_pattern = await self._update_existing_pattern(
                    existing_pattern, pattern_features, execution_result
                )
                discovered_patterns.append(updated_pattern)
            else:
                # Create new pattern
                new_pattern = await self._create_new_pattern(
                    agent_id, pattern_features, pattern_type, execution_result, attention_data
                )
                discovered_patterns.append(new_pattern)
            
            # Update attention-success correlations
            await self._update_attention_correlations(attention_data, execution_result)
            
            # Learn failure predictors
            if not execution_result.get('success', False):
                await self._learn_failure_predictors(pattern_features, execution_result)
            
            # Adapt verification criteria
            await self._adapt_verification_criteria(pattern_features, execution_result)
            
            # Update metrics
            self.engine_metrics['patterns_discovered'] += len(discovered_patterns)
            
            return discovered_patterns
            
        except Exception as e:
            logger.error(f"Error learning from execution: {e}")
            return []

    def _extract_pattern_features(self, task_data: Dict[str, Any],
                                execution_result: Dict[str, Any],
                                attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features for pattern recognition"""
        features = {
            # Task characteristics
            'task_complexity': self._calculate_task_complexity(task_data),
            'task_type': task_data.get('task_type', 'unknown'),
            'task_duration': execution_result.get('execution_time', 0),
            
            # Execution characteristics
            'success': execution_result.get('success', False),
            'confidence': execution_result.get('confidence', 0.5),
            'phases_completed': len(execution_result.get('phases', {})),
            'errors_encountered': len(execution_result.get('errors', [])),
            
            # Attention characteristics
            'attention_allocated': attention_data.get('total_attention', 0),
            'attention_efficiency': attention_data.get('efficiency', 0.5),
            'attention_distribution': attention_data.get('distribution', {}),
            
            # Context factors
            'agent_experience': task_data.get('agent_experience', 0),
            'system_load': task_data.get('system_load', 0.5),
            'time_of_day': time.time() % 86400,  # Seconds in day
            
            # Performance indicators
            'recovery_attempts': execution_result.get('recovery_attempts', 0),
            'optimization_applied': len(execution_result.get('optimizations', [])),
            'collaboration_score': execution_result.get('collaboration_score', 0.5)
        }
        
        return features

    def _classify_pattern_type(self, features: Dict[str, Any]) -> str:
        """Classify the type of behavioral pattern"""
        if features['success'] and features['attention_efficiency'] > 0.7:
            return 'high_efficiency_success'
        elif features['success'] and features['attention_allocated'] < 2.0:
            return 'low_attention_success'
        elif not features['success'] and features['errors_encountered'] > 2:
            return 'error_prone_failure'
        elif not features['success'] and features['attention_allocated'] > 5.0:
            return 'high_attention_failure'
        elif features['recovery_attempts'] > 0 and features['success']:
            return 'recovery_success'
        elif features['collaboration_score'] > 0.7:
            return 'collaborative_pattern'
        else:
            return 'general_pattern'

    def _calculate_task_complexity(self, task_data: Dict[str, Any]) -> float:
        """Calculate task complexity score"""
        complexity_factors = {
            'instruction_length': len(task_data.get('instruction', '')),
            'context_size': len(str(task_data.get('context', {}))),
            'requirements_count': len(task_data.get('requirements', [])),
            'expected_duration': task_data.get('expected_duration', 30.0)
        }
        
        # Normalize and combine factors
        normalized_length = min(1.0, complexity_factors['instruction_length'] / 200)
        normalized_context = min(1.0, complexity_factors['context_size'] / 1000)
        normalized_requirements = min(1.0, complexity_factors['requirements_count'] / 10)
        normalized_duration = min(1.0, complexity_factors['expected_duration'] / 120.0)
        
        complexity = (normalized_length + normalized_context + 
                     normalized_requirements + normalized_duration) / 4.0
        
        return complexity

    def _find_similar_pattern(self, features: Dict[str, Any], pattern_type: str) -> Optional[BehavioralPattern]:
        """Find existing similar pattern"""
        for pattern in self.behavioral_patterns.values():
            if pattern.pattern_type == pattern_type:
                similarity = self._calculate_pattern_similarity(features, pattern.context_factors)
                if similarity > 0.8:  # High similarity threshold
                    return pattern
        return None

    def _calculate_pattern_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two sets of features"""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1e-6)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarity_scores.append(max(0, sim))
            elif val1 == val2:
                # Exact match
                similarity_scores.append(1.0)
            else:
                # No match
                similarity_scores.append(0.0)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0

    async def _create_new_pattern(self, agent_id: str, features: Dict[str, Any],
                                pattern_type: str, execution_result: Dict[str, Any],
                                attention_data: Dict[str, Any]) -> BehavioralPattern:
        """Create new behavioral pattern"""
        pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
        
        pattern = BehavioralPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            success_rate=1.0 if execution_result.get('success', False) else 0.0,
            failure_rate=0.0 if execution_result.get('success', False) else 1.0,
            attention_cost=attention_data.get('total_attention', 0),
            confidence=0.5,  # Initial confidence
            frequency=1,
            last_seen=time.time(),
            context_factors=features.copy(),
            learning_metadata={
                'agent_id': agent_id,
                'created_at': time.time(),
                'sample_size': 1
            }
        )
        
        self.behavioral_patterns[pattern_id] = pattern
        self.pattern_frequency[pattern_type] += 1
        
        logger.info(f"Created new behavioral pattern: {pattern_type}")
        return pattern

    async def _update_existing_pattern(self, pattern: BehavioralPattern,
                                     features: Dict[str, Any],
                                     execution_result: Dict[str, Any]) -> BehavioralPattern:
        """Update existing behavioral pattern with new data"""
        pattern.frequency += 1
        pattern.last_seen = time.time()
        
        # Update success/failure rates
        success = execution_result.get('success', False)
        total_samples = pattern.learning_metadata.get('sample_size', 1) + 1
        
        if success:
            pattern.success_rate = (pattern.success_rate * (total_samples - 1) + 1.0) / total_samples
        else:
            pattern.failure_rate = (pattern.failure_rate * (total_samples - 1) + 1.0) / total_samples
        
        # Update confidence based on sample size
        pattern.confidence = min(0.95, 0.5 + 0.45 * (total_samples / 100.0))
        
        # Update learning metadata
        pattern.learning_metadata['sample_size'] = total_samples
        pattern.learning_metadata['last_updated'] = time.time()
        
        # Adapt context factors (weighted average)
        weight = 0.1  # Learning rate
        for key in features:
            if key in pattern.context_factors:
                if isinstance(features[key], (int, float)):
                    old_val = pattern.context_factors[key]
                    pattern.context_factors[key] = old_val * (1 - weight) + features[key] * weight
        
        logger.debug(f"Updated pattern {pattern.pattern_id}: success_rate={pattern.success_rate:.2f}")
        return pattern

    async def _update_attention_correlations(self, attention_data: Dict[str, Any],
                                           execution_result: Dict[str, Any]) -> None:
        """Update attention-success correlations"""
        attention_cost = attention_data.get('total_attention', 0)
        success = execution_result.get('success', False)
        
        # Find appropriate attention range
        attention_range = self._get_attention_range(attention_cost)
        range_key = f"{attention_range[0]:.1f}-{attention_range[1]:.1f}"
        
        if range_key in self.attention_correlations:
            correlation = self.attention_correlations[range_key]
            
            # Update correlation
            total_samples = correlation.sample_size + 1
            success_count = correlation.success_probability * correlation.sample_size
            if success:
                success_count += 1
            
            correlation.success_probability = success_count / total_samples
            correlation.sample_size = total_samples
            
            # Update optimal attention (weighted average)
            if success:
                weight = 0.1
                correlation.optimal_attention = (
                    correlation.optimal_attention * (1 - weight) + attention_cost * weight
                )
            
            # Update efficiency score
            correlation.efficiency_score = correlation.success_probability / max(0.1, attention_cost)
            
        else:
            # Create new correlation
            self.attention_correlations[range_key] = AttentionSuccessCorrelation(
                attention_range=attention_range,
                success_probability=1.0 if success else 0.0,
                sample_size=1,
                optimal_attention=attention_cost,
                efficiency_score=(1.0 if success else 0.0) / max(0.1, attention_cost)
            )

    def _get_attention_range(self, attention_cost: float) -> Tuple[float, float]:
        """Get attention range bucket for given cost"""
        ranges = [
            (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0),
            (5.0, 8.0), (8.0, 12.0), (12.0, float('inf'))
        ]
        
        for range_tuple in ranges:
            if range_tuple[0] <= attention_cost < range_tuple[1]:
                return range_tuple
        
        return ranges[-1]  # Default to highest range

    async def _learn_failure_predictors(self, features: Dict[str, Any],
                                      execution_result: Dict[str, Any]) -> None:
        """Learn patterns that predict failures"""
        failure_indicators = {
            'high_complexity_low_attention': (
                features['task_complexity'] > 0.7 and features['attention_allocated'] < 2.0
            ),
            'high_error_rate': features['errors_encountered'] > 2,
            'low_confidence': features['confidence'] < 0.3,
            'high_system_load': features['system_load'] > 0.8,
            'insufficient_experience': (
                features['task_complexity'] > 0.5 and features['agent_experience'] < 3
            )
        }
        
        for indicator, condition in failure_indicators.items():
            if condition:
                if indicator not in self.failure_predictors:
                    self.failure_predictors[indicator] = {
                        'prediction_count': 0,
                        'actual_failures': 0,
                        'accuracy': 0.0,
                        'confidence': 0.5
                    }
                
                predictor = self.failure_predictors[indicator]
                predictor['prediction_count'] += 1
                predictor['actual_failures'] += 1  # This was a failure
                predictor['accuracy'] = predictor['actual_failures'] / predictor['prediction_count']
                predictor['confidence'] = min(0.95, 0.5 + 0.45 * (predictor['prediction_count'] / 50))

    async def _adapt_verification_criteria(self, features: Dict[str, Any],
                                         execution_result: Dict[str, Any]) -> None:
        """Adapt verification criteria based on learned patterns"""
        task_type = features['task_type']
        success = execution_result.get('success', False)
        
        if task_type not in self.verification_criteria:
            self.verification_criteria[task_type] = {
                'confidence_threshold': 0.7,
                'attention_threshold': 3.0,
                'timeout_multiplier': 1.0,
                'retry_count': 1,
                'success_samples': 0,
                'total_samples': 0
            }
        
        criteria = self.verification_criteria[task_type]
        criteria['total_samples'] += 1
        
        if success:
            criteria['success_samples'] += 1
            
            # Relax criteria if consistently successful
            success_rate = criteria['success_samples'] / criteria['total_samples']
            if success_rate > 0.9 and criteria['total_samples'] > 10:
                criteria['confidence_threshold'] = max(0.5, criteria['confidence_threshold'] - 0.05)
                criteria['attention_threshold'] = max(1.0, criteria['attention_threshold'] - 0.2)
        else:
            # Tighten criteria if failures occur
            criteria['confidence_threshold'] = min(0.9, criteria['confidence_threshold'] + 0.05)
            criteria['attention_threshold'] = min(8.0, criteria['attention_threshold'] + 0.3)
            criteria['retry_count'] = min(5, criteria['retry_count'] + 1)
        
        # Record adaptation
        self.criteria_adaptation_history.append({
            'timestamp': time.time(),
            'task_type': task_type,
            'criteria': criteria.copy(),
            'trigger': 'success' if success else 'failure'
        })
        
        self.engine_metrics['verification_adaptations'] += 1

    async def predict_task_outcome(self, agent_id: str, task_data: Dict[str, Any],
                                 attention_allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Predict task outcome based on learned patterns"""
        try:
            # Extract features for prediction
            prediction_features = self._extract_prediction_features(task_data, attention_allocation)
            
            # Find matching patterns
            matching_patterns = self._find_matching_patterns(prediction_features)
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(matching_patterns, prediction_features)
            
            # Predict failure points
            failure_predictions = self._predict_failure_points(prediction_features)
            
            # Get optimal attention recommendation
            attention_recommendation = self._recommend_optimal_attention(prediction_features)
            
            # Compile prediction
            prediction = {
                'success_probability': success_probability,
                'confidence': self._calculate_prediction_confidence(matching_patterns),
                'failure_predictions': failure_predictions,
                'attention_recommendation': attention_recommendation,
                'matching_patterns': len(matching_patterns),
                'prediction_timestamp': time.time()
            }
            
            self.engine_metrics['predictions_made'] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting task outcome: {e}")
            return {
                'success_probability': 0.5,
                'confidence': 0.1,
                'error': str(e)
            }

    def _extract_prediction_features(self, task_data: Dict[str, Any],
                                   attention_allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for prediction"""
        return {
            'task_complexity': self._calculate_task_complexity(task_data),
            'task_type': task_data.get('task_type', 'unknown'),
            'expected_duration': task_data.get('expected_duration', 30.0),
            'attention_allocated': attention_allocation.get('total_attention', 0),
            'attention_distribution': attention_allocation.get('distribution', {}),
            'agent_experience': task_data.get('agent_experience', 0),
            'system_load': task_data.get('system_load', 0.5)
        }

    def _find_matching_patterns(self, features: Dict[str, Any]) -> List[BehavioralPattern]:
        """Find patterns that match current features"""
        matching_patterns = []
        
        for pattern in self.behavioral_patterns.values():
            similarity = self._calculate_pattern_similarity(features, pattern.context_factors)
            if similarity > 0.6:  # Similarity threshold for matching
                matching_patterns.append(pattern)
        
        # Sort by similarity and frequency
        matching_patterns.sort(
            key=lambda p: (
                self._calculate_pattern_similarity(features, p.context_factors) * 
                p.frequency * p.confidence
            ),
            reverse=True
        )
        
        return matching_patterns[:10]  # Top 10 matches

    def _calculate_success_probability(self, patterns: List[BehavioralPattern],
                                     features: Dict[str, Any]) -> float:
        """Calculate success probability based on matching patterns"""
        if not patterns:
            return 0.5  # Default probability
        
        # Weighted average based on pattern confidence and similarity
        total_weight = 0
        weighted_success = 0
        
        for pattern in patterns:
            similarity = self._calculate_pattern_similarity(features, pattern.context_factors)
            weight = pattern.confidence * similarity * pattern.frequency
            
            weighted_success += pattern.success_rate * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_success / total_weight
        else:
            return 0.5

    def _predict_failure_points(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict potential failure points"""
        failure_points = []
        
        for indicator, predictor in self.failure_predictors.items():
            if predictor['accuracy'] > 0.6 and predictor['confidence'] > 0.7:
                # Check if failure condition applies
                if self._check_failure_condition(indicator, features):
                    failure_points.append({
                        'type': indicator,
                        'probability': predictor['accuracy'],
                        'confidence': predictor['confidence'],
                        'recommendation': self._get_failure_mitigation(indicator)
                    })
        
        return failure_points

    def _check_failure_condition(self, indicator: str, features: Dict[str, Any]) -> bool:
        """Check if failure condition applies to current features"""
        conditions = {
            'high_complexity_low_attention': (
                features['task_complexity'] > 0.7 and features['attention_allocated'] < 2.0
            ),
            'low_confidence': features.get('confidence', 0.5) < 0.3,
            'high_system_load': features['system_load'] > 0.8,
            'insufficient_experience': (
                features['task_complexity'] > 0.5 and features['agent_experience'] < 3
            )
        }
        
        return conditions.get(indicator, False)

    def _get_failure_mitigation(self, indicator: str) -> str:
        """Get mitigation recommendation for failure indicator"""
        mitigations = {
            'high_complexity_low_attention': 'Increase attention allocation for complex tasks',
            'low_confidence': 'Add verification steps and reduce complexity',
            'high_system_load': 'Delay execution or reduce concurrent tasks',
            'insufficient_experience': 'Provide additional guidance or simplify task'
        }
        
        return mitigations.get(indicator, 'General risk mitigation needed')

    def _recommend_optimal_attention(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal attention allocation"""
        task_type = features['task_type']
        task_complexity = features['task_complexity']
        
        # Find best attention correlation for this context
        best_correlation = None
        best_efficiency = 0
        
        for correlation in self.attention_correlations.values():
            if correlation.efficiency_score > best_efficiency and correlation.sample_size > 5:
                best_correlation = correlation
                best_efficiency = correlation.efficiency_score
        
        if best_correlation:
            recommended_attention = best_correlation.optimal_attention
        else:
            # Fallback: base recommendation on complexity
            recommended_attention = 1.0 + task_complexity * 3.0
        
        return {
            'recommended_attention': recommended_attention,
            'efficiency_score': best_efficiency,
            'confidence': 0.8 if best_correlation else 0.5
        }

    def _calculate_prediction_confidence(self, patterns: List[BehavioralPattern]) -> float:
        """Calculate confidence in prediction"""
        if not patterns:
            return 0.1
        
        # Base confidence on pattern strength and quantity
        avg_confidence = np.mean([p.confidence for p in patterns])
        pattern_count_factor = min(1.0, len(patterns) / 5.0)
        
        return avg_confidence * pattern_count_factor

    def get_engine_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the behavioral pattern engine"""
        # Calculate current metrics
        total_patterns = len(self.behavioral_patterns)
        successful_patterns = sum(1 for p in self.behavioral_patterns.values() if p.success_rate > 0.7)
        
        # Pattern type distribution
        pattern_types = defaultdict(int)
        for pattern in self.behavioral_patterns.values():
            pattern_types[pattern.pattern_type] += 1
        
        # Attention efficiency analysis
        attention_ranges = {}
        for range_key, correlation in self.attention_correlations.items():
            attention_ranges[range_key] = {
                'success_rate': correlation.success_probability,
                'efficiency': correlation.efficiency_score,
                'samples': correlation.sample_size
            }
        
        return {
            'engine_metrics': self.engine_metrics.copy(),
            'pattern_summary': {
                'total_patterns': total_patterns,
                'successful_patterns': successful_patterns,
                'pattern_types': dict(pattern_types),
                'avg_confidence': np.mean([p.confidence for p in self.behavioral_patterns.values()]) if total_patterns > 0 else 0
            },
            'attention_analysis': attention_ranges,
            'failure_predictors': {
                name: {'accuracy': pred['accuracy'], 'confidence': pred['confidence']}
                for name, pred in self.failure_predictors.items()
            },
            'verification_criteria': self.verification_criteria.copy()
        }


__all__ = [
    "BehavioralPatternEngine",
    "BehavioralPattern", 
    "AttentionSuccessCorrelation"
]