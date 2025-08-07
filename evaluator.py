"""
System Evaluator - Performance Analysis and Quality Assessment
Comprehensive evaluation of AMAPI system performance and capabilities
"""

import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from loguru import logger

from core.logger import AMAPILogger, LogCategory


class EvaluationCategory(Enum):
    """Categories of system evaluation"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    USABILITY = "usability"


class EvaluationLevel(Enum):
    """Levels of evaluation depth"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"


@dataclass
class EvaluationMetric:
    """Individual evaluation metric"""
    name: str
    category: EvaluationCategory
    value: float
    max_value: float
    unit: str
    description: str
    timestamp: float
    confidence: float = 1.0


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    evaluation_id: str
    timestamp: float
    level: EvaluationLevel
    overall_score: float
    category_scores: Dict[str, float]
    metrics: List[EvaluationMetric]
    recommendations: List[str]
    issues: List[str]
    strengths: List[str]
    improvement_opportunities: List[str]
    next_evaluation_time: float


class SystemEvaluator:
    """
    Comprehensive System Evaluator
    Analyzes and evaluates AMAPI system performance across multiple dimensions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = AMAPILogger("SystemEvaluator")
        
        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        self.max_history_size = self.config.get('max_history_size', 100)
        
        # Evaluation settings
        self.evaluation_interval = self.config.get('evaluation_interval', 300.0)  # 5 minutes
        self.default_level = EvaluationLevel(self.config.get('default_level', 'standard'))
        
        # Performance baselines
        self.performance_baselines = {
            'task_success_rate': 85.0,
            'average_response_time': 5.0,
            'attention_efficiency': 75.0,
            'system_uptime': 99.0,
            'error_rate': 5.0,
            'resource_utilization': 70.0
        }
        
        # Scoring weights for different categories
        self.category_weights = {
            EvaluationCategory.PERFORMANCE: 0.25,
            EvaluationCategory.RELIABILITY: 0.25,
            EvaluationCategory.EFFICIENCY: 0.20,
            EvaluationCategory.ACCURACY: 0.15,
            EvaluationCategory.SCALABILITY: 0.10,
            EvaluationCategory.USABILITY: 0.05
        }
        
        self.logger.info("System Evaluator initialized")

    async def evaluate_system(self, level: EvaluationLevel = None,
                            target_components: List[str] = None) -> EvaluationResult:
        """Perform comprehensive system evaluation"""
        try:
            evaluation_level = level or self.default_level
            evaluation_id = f"eval_{int(time.time())}_{hash(str(target_components)) % 1000}"
            
            self.logger.info(f"Starting system evaluation: {evaluation_id} (level: {evaluation_level.value})")
            
            # Collect metrics based on evaluation level
            metrics = await self._collect_evaluation_metrics(evaluation_level, target_components)
            
            # Calculate category scores
            category_scores = await self._calculate_category_scores(metrics)
            
            # Calculate overall score
            overall_score = await self._calculate_overall_score(category_scores)
            
            # Generate recommendations and analysis
            recommendations = await self._generate_recommendations(metrics, category_scores)
            issues = await self._identify_issues(metrics, category_scores)
            strengths = await self._identify_strengths(metrics, category_scores)
            improvement_opportunities = await self._identify_improvements(metrics, category_scores)
            
            # Create evaluation result
            evaluation_result = EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=time.time(),
                level=evaluation_level,
                overall_score=overall_score,
                category_scores=category_scores,
                metrics=metrics,
                recommendations=recommendations,
                issues=issues,
                strengths=strengths,
                improvement_opportunities=improvement_opportunities,
                next_evaluation_time=time.time() + self.evaluation_interval
            )
            
            # Store in history
            self.evaluation_history.append(evaluation_result)
            if len(self.evaluation_history) > self.max_history_size:
                self.evaluation_history.pop(0)
            
            self.logger.info(f"System evaluation completed: {evaluation_id} (score: {overall_score:.2f})")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error during system evaluation: {e}")
            raise

    async def _collect_evaluation_metrics(self, level: EvaluationLevel,
                                        target_components: List[str] = None) -> List[EvaluationMetric]:
        """Collect metrics based on evaluation level"""
        try:
            metrics = []
            
            # Basic metrics (always collected)
            basic_metrics = await self._collect_basic_metrics()
            metrics.extend(basic_metrics)
            
            if level in [EvaluationLevel.STANDARD, EvaluationLevel.COMPREHENSIVE, EvaluationLevel.DEEP]:
                standard_metrics = await self._collect_standard_metrics()
                metrics.extend(standard_metrics)
            
            if level in [EvaluationLevel.COMPREHENSIVE, EvaluationLevel.DEEP]:
                comprehensive_metrics = await self._collect_comprehensive_metrics()
                metrics.extend(comprehensive_metrics)
            
            if level == EvaluationLevel.DEEP:
                deep_metrics = await self._collect_deep_metrics()
                metrics.extend(deep_metrics)
            
            # Filter by target components if specified
            if target_components:
                metrics = [
                    metric for metric in metrics
                    if any(component in metric.name.lower() for component in target_components)
                ]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting evaluation metrics: {e}")
            return []

    async def _collect_basic_metrics(self) -> List[EvaluationMetric]:
        """Collect basic system metrics"""
        metrics = []
        
        try:
            # System uptime
            metrics.append(EvaluationMetric(
                name="system_uptime",
                category=EvaluationCategory.RELIABILITY,
                value=99.5,  # Placeholder - would get from actual system
                max_value=100.0,
                unit="percent",
                description="System availability percentage",
                timestamp=time.time()
            ))
            
            # Task success rate
            metrics.append(EvaluationMetric(
                name="task_success_rate",
                category=EvaluationCategory.PERFORMANCE,
                value=87.5,  # Placeholder
                max_value=100.0,
                unit="percent",
                description="Percentage of successfully completed tasks",
                timestamp=time.time()
            ))
            
            # Average response time
            metrics.append(EvaluationMetric(
                name="average_response_time",
                category=EvaluationCategory.PERFORMANCE,
                value=3.2,  # Placeholder
                max_value=10.0,
                unit="seconds",
                description="Average system response time",
                timestamp=time.time()
            ))
            
            # Error rate
            metrics.append(EvaluationMetric(
                name="error_rate",
                category=EvaluationCategory.RELIABILITY,
                value=2.1,  # Placeholder
                max_value=10.0,
                unit="percent",
                description="System error rate",
                timestamp=time.time()
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting basic metrics: {e}")
        
        return metrics

    async def _collect_standard_metrics(self) -> List[EvaluationMetric]:
        """Collect standard system metrics"""
        metrics = []
        
        try:
            # Attention efficiency
            metrics.append(EvaluationMetric(
                name="attention_efficiency",
                category=EvaluationCategory.EFFICIENCY,
                value=78.3,  # Placeholder
                max_value=100.0,
                unit="percent",
                description="Attention allocation efficiency",
                timestamp=time.time()
            ))
            
            # Resource utilization
            metrics.append(EvaluationMetric(
                name="resource_utilization",
                category=EvaluationCategory.EFFICIENCY,
                value=65.4,  # Placeholder
                max_value=100.0,
                unit="percent",
                description="System resource utilization",
                timestamp=time.time()
            ))
            
            # Learning rate
            metrics.append(EvaluationMetric(
                name="learning_rate",
                category=EvaluationCategory.ACCURACY,
                value=0.85,  # Placeholder
                max_value=1.0,
                unit="rate",
                description="Behavioral learning adaptation rate",
                timestamp=time.time()
            ))
            
            # Device compatibility
            metrics.append(EvaluationMetric(
                name="device_compatibility",
                category=EvaluationCategory.SCALABILITY,
                value=92.1,  # Placeholder
                max_value=100.0,
                unit="percent",
                description="Device abstraction compatibility score",
                timestamp=time.time()
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting standard metrics: {e}")
        
        return metrics

    async def _collect_comprehensive_metrics(self) -> List[EvaluationMetric]:
        """Collect comprehensive system metrics"""
        metrics = []
        
        try:
            # Agent coordination efficiency
            metrics.append(EvaluationMetric(
                name="coordination_efficiency",
                category=EvaluationCategory.PERFORMANCE,
                value=82.7,  # Placeholder
                max_value=100.0,
                unit="percent",
                description="Multi-agent coordination efficiency",
                timestamp=time.time()
            ))
            
            # LLM integration quality
            metrics.append(EvaluationMetric(
                name="llm_integration_quality",
                category=EvaluationCategory.ACCURACY,
                value=89.3,  # Placeholder
                max_value=100.0,
                unit="percent",
                description="LLM integration effectiveness",
                timestamp=time.time()
            ))
            
            # Scalability index
            metrics.append(EvaluationMetric(
                name="scalability_index",
                category=EvaluationCategory.SCALABILITY,
                value=7.8,  # Placeholder
                max_value=10.0,
                unit="index",
                description="System scalability assessment",
                timestamp=time.time()
            ))
            
            # User experience score
            metrics.append(EvaluationMetric(
                name="user_experience_score",
                category=EvaluationCategory.USABILITY,
                value=8.2,  # Placeholder
                max_value=10.0,
                unit="score",
                description="Overall user experience rating",
                timestamp=time.time()
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting comprehensive metrics: {e}")
        
        return metrics

    async def _collect_deep_metrics(self) -> List[EvaluationMetric]:
        """Collect deep analysis metrics"""
        metrics = []
        
        try:
            # Pattern recognition accuracy
            metrics.append(EvaluationMetric(
                name="pattern_recognition_accuracy",
                category=EvaluationCategory.ACCURACY,
                value=91.5,  # Placeholder
                max_value=100.0,
                unit="percent",
                description="Behavioral pattern recognition accuracy",
                timestamp=time.time()
            ))
            
            # Adaptation speed
            metrics.append(EvaluationMetric(
                name="adaptation_speed",
                category=EvaluationCategory.EFFICIENCY,
                value=0.73,  # Placeholder
                max_value=1.0,
                unit="rate",
                description="System adaptation speed to new conditions",
                timestamp=time.time()
            ))
            
            # Fault tolerance
            metrics.append(EvaluationMetric(
                name="fault_tolerance",
                category=EvaluationCategory.RELIABILITY,
                value=8.6,  # Placeholder
                max_value=10.0,
                unit="score",
                description="System fault tolerance rating",
                timestamp=time.time()
            ))
            
            # Innovation index
            metrics.append(EvaluationMetric(
                name="innovation_index",
                category=EvaluationCategory.ACCURACY,
                value=7.3,  # Placeholder
                max_value=10.0,
                unit="index",
                description="System innovation and creativity index",
                timestamp=time.time()
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting deep metrics: {e}")
        
        return metrics

    async def _calculate_category_scores(self, metrics: List[EvaluationMetric]) -> Dict[str, float]:
        """Calculate scores for each evaluation category"""
        try:
            category_scores = {}
            
            for category in EvaluationCategory:
                category_metrics = [m for m in metrics if m.category == category]
                
                if category_metrics:
                    # Calculate weighted average based on confidence
                    total_weighted_score = sum(
                        (m.value / m.max_value * 100) * m.confidence for m in category_metrics
                    )
                    total_weight = sum(m.confidence for m in category_metrics)
                    
                    category_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
                    category_scores[category.value] = category_score
                else:
                    category_scores[category.value] = 0.0
            
            return category_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating category scores: {e}")
            return {}

    async def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate overall system score"""
        try:
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for category, weight in self.category_weights.items():
                if category.value in category_scores:
                    total_weighted_score += category_scores[category.value] * weight
                    total_weight += weight
            
            overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            
            return round(overall_score, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.0

    async def _generate_recommendations(self, metrics: List[EvaluationMetric],
                                      category_scores: Dict[str, float]) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        try:
            # Performance recommendations
            if category_scores.get('performance', 0) < 80:
                recommendations.append("Consider optimizing task execution workflows")
                recommendations.append("Review agent coordination patterns for efficiency gains")
            
            # Reliability recommendations
            if category_scores.get('reliability', 0) < 85:
                recommendations.append("Implement additional error handling mechanisms")
                recommendations.append("Enhance system monitoring and alerting")
            
            # Efficiency recommendations
            if category_scores.get('efficiency', 0) < 75:
                recommendations.append("Optimize attention allocation algorithms")
                recommendations.append("Review resource utilization patterns")
            
            # Accuracy recommendations
            if category_scores.get('accuracy', 0) < 90:
                recommendations.append("Improve behavioral pattern recognition")
                recommendations.append("Enhance LLM integration quality")
            
            # Scalability recommendations
            if category_scores.get('scalability', 0) < 80:
                recommendations.append("Design for better horizontal scaling")
                recommendations.append("Optimize device abstraction layer")
            
            # General recommendations based on metrics
            for metric in metrics:
                normalized_score = (metric.value / metric.max_value) * 100
                
                if normalized_score < 70:
                    recommendations.append(f"Address low performance in {metric.name}: {normalized_score:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return recommendations[:10]  # Limit to top 10 recommendations

    async def _identify_issues(self, metrics: List[EvaluationMetric],
                             category_scores: Dict[str, float]) -> List[str]:
        """Identify critical system issues"""
        issues = []
        
        try:
            # Critical performance issues
            for metric in metrics:
                normalized_score = (metric.value / metric.max_value) * 100
                
                if normalized_score < 50:
                    issues.append(f"Critical: {metric.name} at {normalized_score:.1f}% - {metric.description}")
                elif normalized_score < 70:
                    issues.append(f"Warning: {metric.name} at {normalized_score:.1f}% - {metric.description}")
            
            # Category-specific issues
            for category, score in category_scores.items():
                if score < 60:
                    issues.append(f"Critical category performance: {category} at {score:.1f}%")
                elif score < 75:
                    issues.append(f"Concerning category performance: {category} at {score:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error identifying issues: {e}")
        
        return issues

    async def _identify_strengths(self, metrics: List[EvaluationMetric],
                                category_scores: Dict[str, float]) -> List[str]:
        """Identify system strengths"""
        strengths = []
        
        try:
            # High-performing metrics
            for metric in metrics:
                normalized_score = (metric.value / metric.max_value) * 100
                
                if normalized_score > 90:
                    strengths.append(f"Excellent {metric.name}: {normalized_score:.1f}%")
                elif normalized_score > 85:
                    strengths.append(f"Strong {metric.name}: {normalized_score:.1f}%")
            
            # Strong categories
            for category, score in category_scores.items():
                if score > 90:
                    strengths.append(f"Excellent {category} performance: {score:.1f}%")
                elif score > 85:
                    strengths.append(f"Strong {category} performance: {score:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error identifying strengths: {e}")
        
        return strengths

    async def _identify_improvements(self, metrics: List[EvaluationMetric],
                                   category_scores: Dict[str, float]) -> List[str]:
        """Identify improvement opportunities"""
        improvements = []
        
        try:
            # Improvement opportunities from metrics
            for metric in metrics:
                normalized_score = (metric.value / metric.max_value) * 100
                
                if 70 <= normalized_score < 85:
                    potential_gain = 100 - normalized_score
                    improvements.append(f"Optimize {metric.name}: potential {potential_gain:.1f}% improvement")
            
            # Category improvement opportunities
            for category, score in category_scores.items():
                if 75 <= score < 90:
                    improvements.append(f"Focus on {category} optimization for significant gains")
            
        except Exception as e:
            self.logger.error(f"Error identifying improvements: {e}")
        
        return improvements

    async def get_evaluation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get evaluation history"""
        try:
            recent_evaluations = self.evaluation_history[-limit:]
            
            return [
                {
                    'evaluation_id': eval_result.evaluation_id,
                    'timestamp': eval_result.timestamp,
                    'level': eval_result.level.value,
                    'overall_score': eval_result.overall_score,
                    'category_scores': eval_result.category_scores,
                    'issues_count': len(eval_result.issues),
                    'recommendations_count': len(eval_result.recommendations)
                }
                for eval_result in recent_evaluations
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting evaluation history: {e}")
            return []

    async def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from evaluation history"""
        try:
            if len(self.evaluation_history) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Extract scores over time
            timestamps = [eval_result.timestamp for eval_result in self.evaluation_history]
            overall_scores = [eval_result.overall_score for eval_result in self.evaluation_history]
            
            # Calculate trends
            trends = {
                'overall_score_trend': {
                    'current': overall_scores[-1],
                    'previous': overall_scores[-2] if len(overall_scores) > 1 else overall_scores[-1],
                    'change': overall_scores[-1] - overall_scores[-2] if len(overall_scores) > 1 else 0,
                    'average': statistics.mean(overall_scores),
                    'best': max(overall_scores),
                    'worst': min(overall_scores)
                },
                'category_trends': {},
                'evaluation_frequency': len(self.evaluation_history) / max(1, (timestamps[-1] - timestamps[0]) / 3600),  # per hour
                'improvement_rate': 0.0
            }
            
            # Category trends
            for category in EvaluationCategory:
                category_scores = []
                for eval_result in self.evaluation_history:
                    if category.value in eval_result.category_scores:
                        category_scores.append(eval_result.category_scores[category.value])
                
                if category_scores:
                    trends['category_trends'][category.value] = {
                        'current': category_scores[-1],
                        'average': statistics.mean(category_scores),
                        'trend': 'improving' if len(category_scores) > 1 and category_scores[-1] > category_scores[-2] else 'stable'
                    }
            
            # Calculate improvement rate
            if len(overall_scores) > 1:
                time_span = timestamps[-1] - timestamps[0]
                score_improvement = overall_scores[-1] - overall_scores[0]
                trends['improvement_rate'] = score_improvement / max(1, time_span / 3600)  # per hour
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {'error': str(e)}

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation system"""
        try:
            return {
                'total_evaluations': len(self.evaluation_history),
                'evaluation_levels_used': list(set(eval_result.level.value for eval_result in self.evaluation_history)),
                'average_overall_score': statistics.mean([eval_result.overall_score for eval_result in self.evaluation_history]) if self.evaluation_history else 0.0,
                'last_evaluation': self.evaluation_history[-1].timestamp if self.evaluation_history else None,
                'next_scheduled_evaluation': self.evaluation_history[-1].next_evaluation_time if self.evaluation_history else None,
                'performance_baselines': self.performance_baselines.copy(),
                'category_weights': {cat.value: weight for cat, weight in self.category_weights.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting evaluation summary: {e}")
            return {'error': str(e)}


__all__ = [
    "SystemEvaluator",
    "EvaluationResult",
    "EvaluationMetric", 
    "EvaluationCategory",
    "EvaluationLevel"
]