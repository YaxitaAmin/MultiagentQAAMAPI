"""
Enhanced System Evaluator - Advanced AMAPI Performance Analysis
Integrates with all AMAPI components for comprehensive system evaluation
"""

import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
from loguru import logger

from core.logger import AMAPILogger, LogCategory
from core.attention_economics import AttentionEconomicsEngine
from core.behavioral_learning import BehavioralPatternEngine
from core.device_abstraction import UniversalDeviceAbstraction
from evaluator import SystemEvaluator, EvaluationResult, EvaluationLevel, EvaluationCategory


class ComponentHealth(Enum):
    """Component health status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ComponentEvaluation:
    """Individual component evaluation"""
    component_name: str
    health_status: ComponentHealth
    performance_score: float
    reliability_score: float
    efficiency_score: float
    issues: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: float


@dataclass
class SystemHealthAssessment:
    """Complete system health assessment"""
    assessment_id: str
    timestamp: float
    overall_score: float
    overall_health: ComponentHealth
    component_evaluations: List[ComponentEvaluation]
    system_metrics: Dict[str, float]
    integration_health: Dict[str, float]
    predictive_insights: List[str]
    maintenance_recommendations: List[str]
    risk_assessment: Dict[str, Any]


class EnhancedSystemEvaluator:
    """
    Enhanced System Evaluator with AMAPI Integration
    Provides deep analysis of all AMAPI components and their interactions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = AMAPILogger("EnhancedSystemEvaluator")
        
        # Component references (injected)
        self.attention_engine: Optional[AttentionEconomicsEngine] = None
        self.behavioral_engine: Optional[BehavioralPatternEngine] = None
        self.device_abstraction: Optional[UniversalDeviceAbstraction] = None
        
        # Base evaluator
        self.base_evaluator = SystemEvaluator(config.get('base_evaluator', {}))
        
        # Health assessment history
        self.health_assessments: List[SystemHealthAssessment] = []
        self.max_assessments = self.config.get('max_assessments', 50)
        
        # Health thresholds
        self.health_thresholds = {
            ComponentHealth.EXCELLENT: 90.0,
            ComponentHealth.GOOD: 80.0,
            ComponentHealth.FAIR: 70.0,
            ComponentHealth.POOR: 60.0,
            ComponentHealth.CRITICAL: 0.0
        }
        
        # Component weights for overall scoring
        self.component_weights = {
            'attention_engine': 0.25,
            'behavioral_engine': 0.20,
            'device_abstraction': 0.15,
            'agent_coordination': 0.20,
            'llm_integration': 0.10,
            'system_infrastructure': 0.10
        }
        
        # Evaluation intervals
        self.health_check_interval = self.config.get('health_check_interval', 300.0)  # 5 minutes
        self.deep_evaluation_interval = self.config.get('deep_evaluation_interval', 3600.0)  # 1 hour
        
        # Background tasks
        self._health_monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("Enhanced System Evaluator initialized")

    async def initialize(self):
        """Initialize the enhanced evaluator"""
        try:
            # Start continuous health monitoring
            self._health_monitoring_task = asyncio.create_task(self._continuous_health_monitoring())
            
            self.logger.info("Enhanced System Evaluator started")
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced evaluator: {e}")
            raise

    def set_component_references(self, attention_engine: AttentionEconomicsEngine = None,
                                behavioral_engine: BehavioralPatternEngine = None,
                                device_abstraction: UniversalDeviceAbstraction = None):
        """Set references to AMAPI components"""
        self.attention_engine = attention_engine
        self.behavioral_engine = behavioral_engine
        self.device_abstraction = device_abstraction
        
        self.logger.info("Component references updated")

    async def evaluate_system_health(self) -> SystemHealthAssessment:
        """Perform comprehensive system health evaluation"""
        try:
            assessment_id = f"health_{int(time.time())}_{hash(str(time.time())) % 1000}"
            
            self.logger.info(f"Starting system health evaluation: {assessment_id}")
            
            # Evaluate individual components
            component_evaluations = await self._evaluate_all_components()
            
            # Calculate system metrics
            system_metrics = await self._calculate_system_metrics(component_evaluations)
            
            # Assess integration health
            integration_health = await self._assess_integration_health()
            
            # Calculate overall score
            overall_score = await self._calculate_overall_health_score(
                component_evaluations, system_metrics, integration_health
            )
            
            # Determine overall health status
            overall_health = self._determine_health_status(overall_score)
            
            # Generate predictive insights
            predictive_insights = await self._generate_predictive_insights(
                component_evaluations, system_metrics
            )
            
            # Generate maintenance recommendations
            maintenance_recommendations = await self._generate_maintenance_recommendations(
                component_evaluations, system_metrics
            )
            
            # Perform risk assessment
            risk_assessment = await self._perform_risk_assessment(
                component_evaluations, system_metrics
            )
            
            # Create health assessment
            assessment = SystemHealthAssessment(
                assessment_id=assessment_id,
                timestamp=time.time(),
                overall_score=overall_score,
                overall_health=overall_health,
                component_evaluations=component_evaluations,
                system_metrics=system_metrics,
                integration_health=integration_health,
                predictive_insights=predictive_insights,
                maintenance_recommendations=maintenance_recommendations,
                risk_assessment=risk_assessment
            )
            
            # Store assessment
            self.health_assessments.append(assessment)
            if len(self.health_assessments) > self.max_assessments:
                self.health_assessments.pop(0)
            
            self.logger.info(f"System health evaluation completed: {assessment_id} (score: {overall_score:.2f})")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error evaluating system health: {e}")
            raise

    async def _evaluate_all_components(self) -> List[ComponentEvaluation]:
        """Evaluate all AMAPI components"""
        component_evaluations = []
        
        try:
            # Attention Engine Evaluation
            if self.attention_engine:
                attention_eval = await self._evaluate_attention_engine()
                component_evaluations.append(attention_eval)
            
            # Behavioral Engine Evaluation
            if self.behavioral_engine:
                behavioral_eval = await self._evaluate_behavioral_engine()
                component_evaluations.append(behavioral_eval)
            
            # Device Abstraction Evaluation
            if self.device_abstraction:
                device_eval = await self._evaluate_device_abstraction()
                component_evaluations.append(device_eval)
            
            # System Infrastructure Evaluation
            infrastructure_eval = await self._evaluate_system_infrastructure()
            component_evaluations.append(infrastructure_eval)
            
            return component_evaluations
            
        except Exception as e:
            self.logger.error(f"Error evaluating components: {e}")
            return []

    async def _evaluate_attention_engine(self) -> ComponentEvaluation:
        """Evaluate attention economics engine"""
        try:
            analytics = self.attention_engine.get_engine_analytics()
            
            # Calculate performance metrics
            allocation_efficiency = analytics.get('allocation_efficiency', 0.8)
            recharge_efficiency = analytics.get('recharge_efficiency', 0.85)
            collaboration_effectiveness = analytics.get('collaboration_effectiveness', 0.75)
            
            performance_score = (allocation_efficiency + recharge_efficiency + collaboration_effectiveness) / 3 * 100
            
            # Reliability metrics
            total_allocations = analytics.get('total_allocations', 1)
            successful_allocations = analytics.get('successful_allocations', 1)
            reliability_score = (successful_allocations / max(1, total_allocations)) * 100
            
            # Efficiency metrics
            average_attention_waste = analytics.get('average_attention_waste', 0.1)
            efficiency_score = max(0, (1.0 - average_attention_waste)) * 100
            
            # Identify issues
            issues = []
            if allocation_efficiency < 0.7:
                issues.append("Low attention allocation efficiency")
            if average_attention_waste > 0.2:
                issues.append("High attention waste detected")
            if analytics.get('agent_count', 0) == 0:
                issues.append("No agents registered for attention management")
            
            # Generate recommendations
            recommendations = []
            if allocation_efficiency < 0.8:
                recommendations.append("Optimize attention allocation algorithms")
            if recharge_efficiency < 0.8:
                recommendations.append("Review attention recharge mechanisms")
            
            # Determine health status
            overall_component_score = (performance_score + reliability_score + efficiency_score) / 3
            health_status = self._determine_health_status(overall_component_score)
            
            return ComponentEvaluation(
                component_name="attention_engine",
                health_status=health_status,
                performance_score=performance_score,
                reliability_score=reliability_score,
                efficiency_score=efficiency_score,
                issues=issues,
                recommendations=recommendations,
                metrics={
                    'allocation_efficiency': allocation_efficiency,
                    'recharge_efficiency': recharge_efficiency,
                    'collaboration_effectiveness': collaboration_effectiveness,
                    'total_allocations': total_allocations,
                    'attention_waste': average_attention_waste
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating attention engine: {e}")
            return self._create_failed_component_evaluation("attention_engine", str(e))

    async def _evaluate_behavioral_engine(self) -> ComponentEvaluation:
        """Evaluate behavioral learning engine"""
        try:
            analytics = self.behavioral_engine.get_learning_analytics()
            
            # Performance metrics
            learning_metrics = analytics.get('learning_metrics', {})
            patterns_learned = learning_metrics.get('patterns_learned', 0)
            successful_applications = learning_metrics.get('successful_applications', 0)
            total_applications = learning_metrics.get('patterns_applied', 1)
            
            application_success_rate = successful_applications / max(1, total_applications)
            performance_score = application_success_rate * 100
            
            # Reliability metrics
            average_pattern_confidence = analytics.get('pattern_statistics', {}).get('average_effectiveness', 0.8)
            reliability_score = average_pattern_confidence * 100
            
            # Efficiency metrics
            learning_efficiency = learning_metrics.get('learning_efficiency', 0.7)
            efficiency_score = learning_efficiency * 100
            
            # Identify issues
            issues = []
            if patterns_learned < 5:
                issues.append("Low number of learned patterns")
            if application_success_rate < 0.7:
                issues.append("Low pattern application success rate")
            if average_pattern_confidence < 0.6:
                issues.append("Low average pattern confidence")
            
            # Generate recommendations
            recommendations = []
            if learning_efficiency < 0.7:
                recommendations.append("Increase learning event frequency")
            if average_pattern_confidence < 0.8:
                recommendations.append("Improve pattern validation mechanisms")
            
            # Determine health status
            overall_component_score = (performance_score + reliability_score + efficiency_score) / 3
            health_status = self._determine_health_status(overall_component_score)
            
            return ComponentEvaluation(
                component_name="behavioral_engine",
                health_status=health_status,
                performance_score=performance_score,
                reliability_score=reliability_score,
                efficiency_score=efficiency_score,
                issues=issues,
                recommendations=recommendations,
                metrics={
                    'patterns_learned': patterns_learned,
                    'application_success_rate': application_success_rate,
                    'average_confidence': average_pattern_confidence,
                    'learning_efficiency': learning_efficiency,
                    'total_insights': analytics.get('insights_generated', 0)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating behavioral engine: {e}")
            return self._create_failed_component_evaluation("behavioral_engine", str(e))

    async def _evaluate_device_abstraction(self) -> ComponentEvaluation:
        """Evaluate device abstraction layer"""
        try:
            analytics = self.device_abstraction.get_abstraction_analytics()
            
            # Performance metrics
            abstraction_metrics = analytics.get('abstraction_metrics', {})
            successful_adaptations = abstraction_metrics.get('successful_adaptations', 0)
            total_translations = abstraction_metrics.get('translations_performed', 1)
            
            adaptation_success_rate = successful_adaptations / max(1, total_translations)
            performance_score = adaptation_success_rate * 100
            
            # Reliability metrics
            average_compatibility = analytics.get('average_device_compatibility', 0.9)
            reliability_score = average_compatibility * 100
            
            # Efficiency metrics
            cache_hit_rate = abstraction_metrics.get('cache_hit_rate', 0.8)
            efficiency_score = cache_hit_rate * 100
            
            # Identify issues
            issues = []
            if adaptation_success_rate < 0.8:
                issues.append("Low device adaptation success rate")
            if average_compatibility < 0.8:
                issues.append("Low average device compatibility")
            if cache_hit_rate < 0.7:
                issues.append("Low translation cache hit rate")
            
            # Generate recommendations
            recommendations = []
            if adaptation_success_rate < 0.9:
                recommendations.append("Improve device adaptation algorithms")
            if cache_hit_rate < 0.8:
                recommendations.append("Optimize translation caching strategy")
            
            # Determine health status
            overall_component_score = (performance_score + reliability_score + efficiency_score) / 3
            health_status = self._determine_health_status(overall_component_score)
            
            return ComponentEvaluation(
                component_name="device_abstraction",
                health_status=health_status,
                performance_score=performance_score,
                reliability_score=reliability_score,
                efficiency_score=efficiency_score,
                issues=issues,
                recommendations=recommendations,
                metrics={
                    'devices_profiled': analytics.get('device_statistics', {}).get('total_devices', 0),
                    'adaptation_success_rate': adaptation_success_rate,
                    'average_compatibility': average_compatibility,
                    'cache_hit_rate': cache_hit_rate,
                    'adaptations_made': abstraction_metrics.get('adaptations_made', 0)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating device abstraction: {e}")
            return self._create_failed_component_evaluation("device_abstraction", str(e))

    async def _evaluate_system_infrastructure(self) -> ComponentEvaluation:
        """Evaluate system infrastructure"""
        try:
            # Simulated infrastructure metrics (in production, would query actual system)
            cpu_usage = 45.0  # percentage
            memory_usage = 60.0  # percentage
            disk_usage = 30.0  # percentage
            network_latency = 15.0  # milliseconds
            
            # Performance score based on resource utilization
            performance_score = max(0, 100 - max(cpu_usage, memory_usage, disk_usage))
            
            # Reliability score based on stability metrics
            uptime_percentage = 99.8  # simulated
            reliability_score = uptime_percentage
            
            # Efficiency score based on resource optimization
            efficiency_score = max(0, 100 - (cpu_usage + memory_usage + disk_usage) / 3)
            
            # Identify issues
            issues = []
            if cpu_usage > 80:
                issues.append("High CPU utilization")
            if memory_usage > 85:
                issues.append("High memory utilization")
            if network_latency > 50:
                issues.append("High network latency")
            
            # Generate recommendations
            recommendations = []
            if cpu_usage > 70:
                recommendations.append("Consider CPU optimization or scaling")
            if memory_usage > 75:
                recommendations.append("Monitor memory usage patterns")
            
            # Determine health status
            overall_component_score = (performance_score + reliability_score + efficiency_score) / 3
            health_status = self._determine_health_status(overall_component_score)
            
            return ComponentEvaluation(
                component_name="system_infrastructure",
                health_status=health_status,
                performance_score=performance_score,
                reliability_score=reliability_score,
                efficiency_score=efficiency_score,
                issues=issues,
                recommendations=recommendations,
                metrics={
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'network_latency': network_latency,
                    'uptime_percentage': uptime_percentage
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating system infrastructure: {e}")
            return self._create_failed_component_evaluation("system_infrastructure", str(e))

    def _create_failed_component_evaluation(self, component_name: str, error: str) -> ComponentEvaluation:
        """Create a failed component evaluation"""
        return ComponentEvaluation(
            component_name=component_name,
            health_status=ComponentHealth.CRITICAL,
            performance_score=0.0,
            reliability_score=0.0,
            efficiency_score=0.0,
            issues=[f"Component evaluation failed: {error}"],
            recommendations=[f"Investigate {component_name} component issues"],
            metrics={'evaluation_error': True},
            timestamp=time.time()
        )

    def _determine_health_status(self, score: float) -> ComponentHealth:
        """Determine health status from score"""
        for status, threshold in self.health_thresholds.items():
            if score >= threshold:
                return status
        return ComponentHealth.CRITICAL

    async def _calculate_system_metrics(self, component_evaluations: List[ComponentEvaluation]) -> Dict[str, float]:
        """Calculate overall system metrics"""
        try:
            if not component_evaluations:
                return {}
            
            # Aggregate component scores
            avg_performance = statistics.mean([comp.performance_score for comp in component_evaluations])
            avg_reliability = statistics.mean([comp.reliability_score for comp in component_evaluations])
            avg_efficiency = statistics.mean([comp.efficiency_score for comp in component_evaluations])
            
            # Count issues and recommendations
            total_issues = sum(len(comp.issues) for comp in component_evaluations)
            total_recommendations = sum(len(comp.recommendations) for comp in component_evaluations)
            
            # Health distribution
            health_distribution = {}
            for health_status in ComponentHealth:
                count = sum(1 for comp in component_evaluations if comp.health_status == health_status)
                health_distribution[health_status.value] = count
            
            return {
                'average_performance_score': avg_performance,
                'average_reliability_score': avg_reliability,
                'average_efficiency_score': avg_efficiency,
                'total_issues': total_issues,
                'total_recommendations': total_recommendations,
                'components_evaluated': len(component_evaluations),
                'healthy_components': health_distribution.get('excellent', 0) + health_distribution.get('good', 0),
                'critical_components': health_distribution.get('critical', 0),
                'health_distribution': health_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating system metrics: {e}")
            return {}

    async def _assess_integration_health(self) -> Dict[str, float]:
        """Assess health of component integrations"""
        try:
            integration_scores = {}
            
            # Attention-Behavioral integration
            if self.attention_engine and self.behavioral_engine:
                integration_scores['attention_behavioral'] = 0.85  # Simulated
            
            # Attention-Device integration
            if self.attention_engine and self.device_abstraction:
                integration_scores['attention_device'] = 0.80  # Simulated
            
            # Behavioral-Device integration
            if self.behavioral_engine and self.device_abstraction:
                integration_scores['behavioral_device'] = 0.90  # Simulated
            
            # Overall integration health
            if integration_scores:
                integration_scores['overall_integration_health'] = statistics.mean(integration_scores.values())
            
            return integration_scores
            
        except Exception as e:
            self.logger.error(f"Error assessing integration health: {e}")
            return {}

    async def _calculate_overall_health_score(self, component_evaluations: List[ComponentEvaluation],
                                            system_metrics: Dict[str, float],
                                            integration_health: Dict[str, float]) -> float:
        """Calculate overall system health score"""
        try:
            if not component_evaluations:
                return 0.0
            
            # Component scores with weights
            weighted_score = 0.0
            total_weight = 0.0
            
            for comp in component_evaluations:
                component_score = (comp.performance_score + comp.reliability_score + comp.efficiency_score) / 3
                weight = self.component_weights.get(comp.component_name, 0.1)
                
                weighted_score += component_score * weight
                total_weight += weight
            
            # Integration health contribution
            integration_score = integration_health.get('overall_integration_health', 0.8) * 100
            weighted_score += integration_score * 0.1
            total_weight += 0.1
            
            # Penalty for critical issues
            critical_components = system_metrics.get('critical_components', 0)
            total_components = system_metrics.get('components_evaluated', 1)
            
            if critical_components > 0:
                penalty = (critical_components / total_components) * 20  # Up to 20% penalty
                weighted_score -= penalty
            
            overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            return max(0.0, min(100.0, overall_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating overall health score: {e}")
            return 0.0

    async def _generate_predictive_insights(self, component_evaluations: List[ComponentEvaluation],
                                          system_metrics: Dict[str, float]) -> List[str]:
        """Generate predictive insights based on current health"""
        insights = []
        
        try:
            # Performance trend analysis
            avg_performance = system_metrics.get('average_performance_score', 80.0)
            if avg_performance < 75:
                insights.append("System performance may degrade further without intervention")
            
            # Reliability predictions
            avg_reliability = system_metrics.get('average_reliability_score', 85.0)
            if avg_reliability < 80:
                insights.append("Increased failure probability in the next 24 hours")
            
            # Component-specific predictions
            for comp in component_evaluations:
                if comp.health_status == ComponentHealth.FAIR:
                    insights.append(f"{comp.component_name} may require attention within 6 hours")
                elif comp.health_status == ComponentHealth.POOR:
                    insights.append(f"{comp.component_name} likely to fail within 2 hours")
                elif comp.health_status == ComponentHealth.CRITICAL:
                    insights.append(f"{comp.component_name} requires immediate attention")
            
            # Resource exhaustion predictions
            for comp in component_evaluations:
                if comp.component_name == "system_infrastructure":
                    cpu_usage = comp.metrics.get('cpu_usage', 0)
                    memory_usage = comp.metrics.get('memory_usage', 0)
                    
                    if cpu_usage > 75:
                        insights.append("CPU resources may be exhausted within 1 hour at current usage rate")
                    if memory_usage > 80:
                        insights.append("Memory resources approaching critical levels")
            
        except Exception as e:
            self.logger.error(f"Error generating predictive insights: {e}")
        
        return insights[:10]  # Limit to top 10 insights

    async def _generate_maintenance_recommendations(self, component_evaluations: List[ComponentEvaluation],
                                                  system_metrics: Dict[str, float]) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []
        
        try:
            # Aggregate component recommendations
            for comp in component_evaluations:
                recommendations.extend(comp.recommendations)
            
            # System-level recommendations
            critical_components = system_metrics.get('critical_components', 0)
            if critical_components > 0:
                recommendations.append("Perform immediate system health check")
                recommendations.append("Consider system restart if safe to do so")
            
            total_issues = system_metrics.get('total_issues', 0)
            if total_issues > 5:
                recommendations.append("Schedule comprehensive system maintenance")
            
            # Resource optimization recommendations
            avg_efficiency = system_metrics.get('average_efficiency_score', 80.0)
            if avg_efficiency < 70:
                recommendations.append("Optimize resource allocation algorithms")
                recommendations.append("Review system configuration parameters")
            
        except Exception as e:
            self.logger.error(f"Error generating maintenance recommendations: {e}")
        
        return recommendations[:15]  # Limit to top 15 recommendations

    async def _perform_risk_assessment(self, component_evaluations: List[ComponentEvaluation],
                                     system_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        try:
            risk_assessment = {
                'overall_risk_level': 'low',
                'risk_factors': [],
                'mitigation_strategies': [],
                'probability_estimates': {},
                'impact_estimates': {}
            }
            
            # Component risk analysis
            high_risk_components = []
            for comp in component_evaluations:
                if comp.health_status in [ComponentHealth.POOR, ComponentHealth.CRITICAL]:
                    high_risk_components.append(comp.component_name)
                    risk_assessment['risk_factors'].append(f"{comp.component_name} in {comp.health_status.value} state")
            
            # Overall risk level determination
            critical_count = system_metrics.get('critical_components', 0)
            total_count = system_metrics.get('components_evaluated', 1)
            
            if critical_count > 0:
                risk_assessment['overall_risk_level'] = 'critical'
            elif len(high_risk_components) > total_count * 0.5:
                risk_assessment['overall_risk_level'] = 'high'
            elif len(high_risk_components) > 0:
                risk_assessment['overall_risk_level'] = 'medium'
            
            # Probability estimates
            avg_performance = system_metrics.get('average_performance_score', 80.0)
            avg_reliability = system_metrics.get('average_reliability_score', 85.0)
            
            risk_assessment['probability_estimates'] = {
                'system_failure_24h': max(0.0, (100 - avg_reliability) / 100 * 0.2),
                'performance_degradation_6h': max(0.0, (100 - avg_performance) / 100 * 0.3),
                'component_failure_12h': len(high_risk_components) / max(1, total_count) * 0.4
            }
            
            # Impact estimates
            risk_assessment['impact_estimates'] = {
                'service_disruption': 'medium' if critical_count == 0 else 'high',
                'data_loss_risk': 'low',
                'recovery_time_estimate': '15-30 minutes' if critical_count == 0 else '1-2 hours'
            }
            
            # Mitigation strategies
            if critical_count > 0:
                risk_assessment['mitigation_strategies'].append("Immediate critical component intervention")
            if len(high_risk_components) > 0:
                risk_assessment['mitigation_strategies'].append("Proactive component maintenance")
            if avg_performance < 70:
                risk_assessment['mitigation_strategies'].append("Performance optimization measures")
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error performing risk assessment: {e}")
            return {'error': str(e)}

    async def _continuous_health_monitoring(self):
        """Continuous health monitoring background task"""
        try:
            while True:
                await asyncio.sleep(self.health_check_interval)
                
                try:
                    # Perform health assessment
                    assessment = await self.evaluate_system_health()
                    
                    # Log critical issues
                    if assessment.overall_health == ComponentHealth.CRITICAL:
                        self.logger.critical(f"SYSTEM CRITICAL: Health score {assessment.overall_score:.2f}")
                    elif assessment.overall_health == ComponentHealth.POOR:
                        self.logger.warning(f"SYSTEM DEGRADED: Health score {assessment.overall_score:.2f}")
                    
                    # Process high-priority alerts
                    await self._process_health_alerts(assessment)
                    
                except Exception as e:
                    self.logger.error(f"Error in health monitoring cycle: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Health monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous health monitoring: {e}")

    async def _process_health_alerts(self, assessment: SystemHealthAssessment):
        """Process health alerts and notifications"""
        try:
            # Critical component alerts
            for comp in assessment.component_evaluations:
                if comp.health_status == ComponentHealth.CRITICAL:
                    self.logger.critical(f"CRITICAL COMPONENT: {comp.component_name}")
                elif comp.health_status == ComponentHealth.POOR:
                    self.logger.warning(f"DEGRADED COMPONENT: {comp.component_name}")
            
            # Risk alerts
            risk_level = assessment.risk_assessment.get('overall_risk_level', 'low')
            if risk_level in ['high', 'critical']:
                self.logger.warning(f"HIGH RISK DETECTED: {risk_level}")
            
        except Exception as e:
            self.logger.error(f"Error processing health alerts: {e}")

    async def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            recent_assessments = [
                assessment for assessment in self.health_assessments
                if assessment.timestamp >= cutoff_time
            ]
            
            if len(recent_assessments) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Overall score trend
            scores = [assessment.overall_score for assessment in recent_assessments]
            timestamps = [assessment.timestamp for assessment in recent_assessments]
            
            # Component health trends
            component_trends = {}
            for comp_name in self.component_weights.keys():
                comp_scores = []
                for assessment in recent_assessments:
                    for comp in assessment.component_evaluations:
                        if comp.component_name == comp_name:
                            comp_score = (comp.performance_score + comp.reliability_score + comp.efficiency_score) / 3
                            comp_scores.append(comp_score)
                            break
                
                if len(comp_scores) > 1:
                    trend = 'improving' if comp_scores[-1] > comp_scores[0] else 'declining' if comp_scores[-1] < comp_scores[0] else 'stable'
                    component_trends[comp_name] = {
                        'trend': trend,
                        'current_score': comp_scores[-1],
                        'change': comp_scores[-1] - comp_scores[0]
                    }
            
            return {
                'time_period_hours': hours,
                'assessments_analyzed': len(recent_assessments),
                'overall_score_trend': {
                    'current': scores[-1],
                    'initial': scores[0],
                    'change': scores[-1] - scores[0],
                    'average': statistics.mean(scores),
                    'trend': 'improving' if scores[-1] > scores[0] else 'declining' if scores[-1] < scores[0] else 'stable'
                },
                'component_trends': component_trends,
                'trend_analysis_timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health trends: {e}")
            return {'error': str(e)}

    def get_latest_assessment(self) -> Optional[SystemHealthAssessment]:
        """Get the most recent health assessment"""
        if self.health_assessments:
            return self.health_assessments[-1]
        return None

    async def cleanup(self):
        """Cleanup enhanced evaluator"""
        try:
            if self._health_monitoring_task:
                self._health_monitoring_task.cancel()
                
            self.logger.info("Enhanced System Evaluator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during enhanced evaluator cleanup: {e}")


__all__ = [
    "EnhancedSystemEvaluator",
    "SystemHealthAssessment",
    "ComponentEvaluation",
    "ComponentHealth"
]