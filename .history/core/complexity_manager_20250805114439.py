"""
Dynamic Task Complexity Manager for AMAPI System
Auto-adjusts task difficulty based on agent performance
"""

import time
import json
import uuid
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from loguru import logger


class ComplexityLevel(Enum):
    """Task complexity levels"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class TaskComplexityProfile:
    """Profile of task complexity characteristics"""
    task_id: str
    base_complexity: float
    adjusted_complexity: float
    complexity_level: ComplexityLevel
    complexity_factors: Dict[str, float]
    performance_requirements: Dict[str, Any]
    adaptation_history: List[Dict[str, Any]]
    success_threshold: float


@dataclass
class AgentCapabilityProfile:
    """Agent capability assessment"""
    agent_id: str
    current_capability: float
    capability_growth_rate: float
    specialization_areas: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    optimal_complexity_range: Tuple[float, float]
    learning_velocity: float


class DynamicComplexityManager:
    """
    Manages dynamic task complexity adjustment based on agent performance
    Implements progressive learning and attention-based scaling
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Complexity management
        self.task_profiles: Dict[str, TaskComplexityProfile] = {}
        self.agent_capabilities: Dict[str, AgentCapabilityProfile] = {}
        self.complexity_adaptation_rules: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.complexity_adjustments: List[Dict[str, Any]] = []
        self.learning_progression: Dict[str, List[float]] = {}
        
        # Attention integration
        self.attention_complexity_correlations: Dict[str, Dict[str, Any]] = {}
        self.attention_efficiency_thresholds: Dict[str, float] = {}
        
        # Manager metrics
        self.manager_metrics = {
            'total_adjustments': 0,
            'successful_adaptations': 0,
            'capability_improvements': 0,
            'optimal_matches': 0,
            'attention_optimizations': 0,
            'progression_rate': 0.0
        }
        
        # Initialize complexity factors
        self._initialize_complexity_factors()
        
        logger.info("Dynamic Complexity Manager initialized")

    def _initialize_complexity_factors(self):
        """Initialize complexity calculation factors"""
        self.complexity_factors = {
            'instruction_complexity': {
                'weight': 0.25,
                'calculation': self._calculate_instruction_complexity
            },
            'context_complexity': {
                'weight': 0.20,
                'calculation': self._calculate_context_complexity
            },
            'interaction_complexity': {
                'weight': 0.25,
                'calculation': self._calculate_interaction_complexity
            },
            'verification_complexity': {
                'weight': 0.15,
                'calculation': self._calculate_verification_complexity
            },
            'attention_demand': {
                'weight': 0.15,
                'calculation': self._calculate_attention_demand
            }
        }

    async def assess_task_complexity(self, task_data: Dict[str, Any]) -> TaskComplexityProfile:
        """Assess and create complexity profile for a task"""
        task_id = task_data.get('task_id', f"task_{uuid.uuid4().hex[:8]}")
        
        try:
            # Calculate base complexity
            complexity_scores = {}
            total_complexity = 0.0
            
            for factor_name, factor_config in self.complexity_factors.items():
                score = factor_config['calculation'](task_data)
                complexity_scores[factor_name] = score
                total_complexity += score * factor_config['weight']
            
            # Determine complexity level
            complexity_level = self._determine_complexity_level(total_complexity)
            
            # Create task profile
            profile = TaskComplexityProfile(
                task_id=task_id,
                base_complexity=total_complexity,
                adjusted_complexity=total_complexity,  # Initially same as base
                complexity_level=complexity_level,
                complexity_factors=complexity_scores,
                performance_requirements=self._generate_performance_requirements(total_complexity),
                adaptation_history=[],
                success_threshold=self._calculate_success_threshold(total_complexity)
            )
            
            self.task_profiles[task_id] = profile
            
            logger.info(f"Task complexity assessed: {complexity_level.value} ({total_complexity:.2f})")
            return profile
            
        except Exception as e:
            logger.error(f"Error assessing task complexity: {e}")
            # Return default profile
            return TaskComplexityProfile(
                task_id=task_id,
                base_complexity=0.5,
                adjusted_complexity=0.5,
                complexity_level=ComplexityLevel.MODERATE,
                complexity_factors={},
                performance_requirements={},
                adaptation_history=[],
                success_threshold=0.7
            )

    def _calculate_instruction_complexity(self, task_data: Dict[str, Any]) -> float:
        """Calculate complexity based on instruction characteristics"""
        instruction = task_data.get('instruction', '')
        
        factors = {
            'length': len(instruction) / 200.0,  # Normalize to ~200 chars
            'word_count': len(instruction.split()) / 30.0,  # Normalize to ~30 words
            'conditional_statements': instruction.count('if') + instruction.count('when') + instruction.count('unless'),
            'sequence_indicators': instruction.count('then') + instruction.count('after') + instruction.count('before'),
            'negations': instruction.count('not') + instruction.count("don't") + instruction.count("won't")
        }
        
        # Combine factors with weights
        complexity = (
            factors['length'] * 0.3 +
            factors['word_count'] * 0.3 +
            factors['conditional_statements'] * 0.15 +
            factors['sequence_indicators'] * 0.15 +
            factors['negations'] * 0.1
        )
        
        return min(1.0, complexity)

    def _calculate_context_complexity(self, task_data: Dict[str, Any]) -> float:
        """Calculate complexity based on context requirements"""
        context = task_data.get('context', {})
        requirements = task_data.get('requirements', [])
        
        factors = {
            'context_size': len(str(context)) / 500.0,  # Normalize to ~500 chars
            'requirement_count': len(requirements) / 10.0,  # Normalize to ~10 requirements
            'nested_context': self._count_nested_levels(context),
            'cross_references': self._count_cross_references(context),
            'dependency_complexity': len(task_data.get('dependencies', [])) / 5.0
        }
        
        complexity = (
            factors['context_size'] * 0.25 +
            factors['requirement_count'] * 0.25 +
            factors['nested_context'] * 0.2 +
            factors['cross_references'] * 0.15 +
            factors['dependency_complexity'] * 0.15
        )
        
        return min(1.0, complexity)

    def _calculate_interaction_complexity(self, task_data: Dict[str, Any]) -> float:
        """Calculate complexity based on required interactions"""
        instruction = task_data.get('instruction', '').lower()
        
        # Define interaction complexity weights
        interaction_patterns = {
            'simple_tap': ['tap', 'click', 'touch'],
            'navigation': ['navigate', 'go to', 'open'],
            'input': ['type', 'enter', 'input'],
            'gesture': ['swipe', 'scroll', 'drag'],
            'verification': ['verify', 'check', 'confirm'],
            'complex_sequence': ['toggle', 'switch', 'configure'],
            'multi_step': ['and then', 'after', 'followed by']
        }
        
        interaction_weights = {
            'simple_tap': 0.1,
            'navigation': 0.2,
            'input': 0.3,
            'gesture': 0.4,
            'verification': 0.3,
            'complex_sequence': 0.6,
            'multi_step': 0.8
        }
        
        complexity = 0.0
        for interaction_type, patterns in interaction_patterns.items():
            for pattern in patterns:
                if pattern in instruction:
                    complexity = max(complexity, interaction_weights[interaction_type])
        
        return complexity

    def _calculate_verification_complexity(self, task_data: Dict[str, Any]) -> float:
        """Calculate complexity based on verification requirements"""
        instruction = task_data.get('instruction', '').lower()
        context = task_data.get('context', {})
        
        verification_indicators = {
            'state_verification': ['verify', 'check', 'ensure', 'confirm'],
            'visual_verification': ['see', 'visible', 'display', 'show'],
            'behavioral_verification': ['works', 'functions', 'responds'],
            'comparative_verification': ['different', 'changed', 'updated'],
            'temporal_verification': ['before', 'after', 'during']
        }
        
        complexity = 0.0
        for category, indicators in verification_indicators.items():
            category_score = sum(1 for indicator in indicators if indicator in instruction)
            complexity += category_score * 0.1
        
        # Additional complexity from verification criteria
        verification_criteria = context.get('verification_criteria', [])
        complexity += len(verification_criteria) * 0.05
        
        return min(1.0, complexity)

    def _calculate_attention_demand(self, task_data: Dict[str, Any]) -> float:
        """Calculate attention demand based on task characteristics"""
        instruction = task_data.get('instruction', '')
        expected_duration = task_data.get('expected_duration', 30.0)
        
        factors = {
            'duration_factor': min(1.0, expected_duration / 120.0),  # Normalize to 2 minutes
            'precision_requirement': len([word for word in instruction.lower().split() 
                                        if word in ['exactly', 'precisely', 'carefully', 'slowly']]) * 0.1,
            'error_sensitivity': len([word for word in instruction.lower().split() 
                                    if word in ['important', 'critical', 'must', 'ensure']]) * 0.1,
            'multitasking': instruction.lower().count('while') + instruction.lower().count('simultaneously')
        }
        
        attention_demand = (
            factors['duration_factor'] * 0.4 +
            factors['precision_requirement'] * 0.3 +
            factors['error_sensitivity'] * 0.2 +
            factors['multitasking'] * 0.1
        )
        
        return min(1.0, attention_demand)

    def _count_nested_levels(self, data: Any, current_level: int = 0) -> float:
        """Count nesting levels in data structure"""
        if isinstance(data, dict):
            if not data:
                return current_level
            return max(self._count_nested_levels(v, current_level + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_level
            return max(self._count_nested_levels(item, current_level + 1) for item in data)
        else:
            return current_level

    def _count_cross_references(self, context: Dict[str, Any]) -> float:
        """Count cross-references in context"""
        context_str = str(context).lower()
        reference_patterns = ['refer to', 'see', 'mentioned', 'above', 'below', 'previous', 'following']
        return sum(context_str.count(pattern) for pattern in reference_patterns) * 0.1

    def _determine_complexity_level(self, complexity_score: float) -> ComplexityLevel:
        """Determine complexity level from score"""
        if complexity_score < 0.15:
            return ComplexityLevel.TRIVIAL
        elif complexity_score < 0.3:
            return ComplexityLevel.SIMPLE
        elif complexity_score < 0.5:
            return ComplexityLevel.MODERATE
        elif complexity_score < 0.7:
            return ComplexityLevel.COMPLEX
        elif complexity_score < 0.85:
            return ComplexityLevel.ADVANCED
        else:
            return ComplexityLevel.EXPERT

    def _generate_performance_requirements(self, complexity: float) -> Dict[str, Any]:
        """Generate performance requirements based on complexity"""
        base_requirements = {
            'min_success_rate': max(0.5, 0.9 - complexity * 0.3),
            'max_execution_time': 30 + complexity * 90,  # 30-120 seconds
            'max_retry_attempts': min(5, int(1 + complexity * 4)),
            'required_confidence': max(0.6, 0.9 - complexity * 0.2),
            'attention_budget': 2.0 + complexity * 6.0  # 2-8 attention units
        }
        
        return base_requirements

    def _calculate_success_threshold(self, complexity: float) -> float:
        """Calculate success threshold based on complexity"""
        return max(0.5, 0.85 - complexity * 0.25)

    async def update_agent_capability(self, agent_id: str, performance_data: Dict[str, Any]) -> None:
        """Update agent capability profile based on performance"""
        try:
            if agent_id not in self.agent_capabilities:
                # Initialize new agent capability profile
                self.agent_capabilities[agent_id] = AgentCapabilityProfile(
                    agent_id=agent_id,
                    current_capability=0.5,  # Start at moderate
                    capability_growth_rate=0.0,
                    specialization_areas={},
                    performance_trends={},
                    optimal_complexity_range=(0.3, 0.7),
                    learning_velocity=0.1
                )
            
            profile = self.agent_capabilities[agent_id]
            
            # Update capability based on recent performance
            success_rate = performance_data.get('success_rate', 0.5)
            task_complexity = performance_data.get('task_complexity', 0.5)
            execution_efficiency = performance_data.get('execution_efficiency', 0.5)
            
            # Calculate capability adjustment
            performance_score = (success_rate + execution_efficiency) / 2.0
            complexity_factor = task_complexity
            
            # Update current capability (weighted moving average)
            weight = 0.1  # Learning rate
            capability_change = (performance_score - 0.5) * complexity_factor * weight
            profile.current_capability = max(0.1, min(1.0, profile.current_capability + capability_change))
            
            # Update growth rate
            if len(self.performance_history) > 1:
                recent_performances = [p for p in self.performance_history[-10:] if p['agent_id'] == agent_id]
                if len(recent_performances) >= 2:
                    growth = recent_performances[-1]['capability'] - recent_performances[0]['capability']
                    profile.capability_growth_rate = growth / len(recent_performances)
            
            # Update specialization areas
            task_type = performance_data.get('task_type', 'general')
            if task_type not in profile.specialization_areas:
                profile.specialization_areas[task_type] = 0.5
            
            area_weight = 0.2
            profile.specialization_areas[task_type] = (
                profile.specialization_areas[task_type] * (1 - area_weight) + 
                performance_score * area_weight
            )
            
            # Update optimal complexity range
            if success_rate > 0.8:
                # Expand upper bound if consistently successful
                upper_bound = min(1.0, profile.optimal_complexity_range[1] + 0.05)
                profile.optimal_complexity_range = (profile.optimal_complexity_range[0], upper_bound)
            elif success_rate < 0.6:
                # Lower upper bound if struggling
                upper_bound = max(0.2, profile.optimal_complexity_range[1] - 0.05)
                profile.optimal_complexity_range = (profile.optimal_complexity_range[0], upper_bound)
            
            # Record performance history
            self.performance_history.append({
                'timestamp': time.time(),
                'agent_id': agent_id,
                'capability': profile.current_capability,
                'performance_score': performance_score,
                'task_complexity': task_complexity
            })
            
            # Update metrics
            if capability_change > 0:
                self.manager_metrics['capability_improvements'] += 1
            
            logger.debug(f"Updated capability for {agent_id}: {profile.current_capability:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating agent capability: {e}")

    async def adjust_task_complexity(self, task_id: str, agent_id: str, 
                                   performance_feedback: Dict[str, Any]) -> float:
        """Adjust task complexity based on agent performance"""
        try:
            if task_id not in self.task_profiles or agent_id not in self.agent_capabilities:
                return 0.5  # Default complexity
            
            task_profile = self.task_profiles[task_id]
            agent_profile = self.agent_capabilities[agent_id]
            
            # Analyze performance feedback
            success = performance_feedback.get('success', False)
            execution_time = performance_feedback.get('execution_time', 0)
            attention_used = performance_feedback.get('attention_used', 0)
            confidence = performance_feedback.get('confidence', 0.5)
            
            # Calculate performance metrics
            expected_time = task_profile.performance_requirements['max_execution_time']
            time_efficiency = min(1.0, expected_time / max(1, execution_time))
            
            expected_attention = task_profile.performance_requirements['attention_budget']
            attention_efficiency = min(1.0, expected_attention / max(0.1, attention_used))
            
            performance_score = (
                (1.0 if success else 0.0) * 0.4 +
                time_efficiency * 0.2 +
                attention_efficiency * 0.2 +
                confidence * 0.2
            )
            
            # Determine adjustment direction and magnitude
            adjustment = 0.0
            
            if performance_score > 0.85:
                # Task too easy - increase complexity
                adjustment = min(0.1, (performance_score - 0.85) * 0.5)
            elif performance_score < 0.6:
                # Task too hard - decrease complexity
                adjustment = -min(0.1, (0.6 - performance_score) * 0.5)
            
            # Apply capability-based scaling
            capability_factor = agent_profile.current_capability
            adjustment *= capability_factor
            
            # Update task complexity
            old_complexity = task_profile.adjusted_complexity
            task_profile.adjusted_complexity = max(0.1, min(1.0, old_complexity + adjustment))
            
            # Record adjustment
            adjustment_record = {
                'timestamp': time.time(),
                'task_id': task_id,
                'agent_id': agent_id,
                'old_complexity': old_complexity,
                'new_complexity': task_profile.adjusted_complexity,
                'adjustment': adjustment,
                'performance_score': performance_score,
                'trigger': 'performance_feedback'
            }
            
            task_profile.adaptation_history.append(adjustment_record)
            self.complexity_adjustments.append(adjustment_record)
            
            # Update metrics
            self.manager_metrics['total_adjustments'] += 1
            if abs(adjustment) > 0.05:  # Significant adjustment
                self.manager_metrics['successful_adaptations'] += 1
            
            # Check if complexity matches agent capability (optimal matching)
            optimal_range = agent_profile.optimal_complexity_range
            if optimal_range[0] <= task_profile.adjusted_complexity <= optimal_range[1]:
                self.manager_metrics['optimal_matches'] += 1
            
            logger.info(f"Adjusted task complexity: {old_complexity:.2f} -> {task_profile.adjusted_complexity:.2f}")
            
            return task_profile.adjusted_complexity
            
        except Exception as e:
            logger.error(f"Error adjusting task complexity: {e}")
            return 0.5

    async def match_task_to_agent_capability(self, task_id: str, 
                                           available_agents: List[str]) -> Dict[str, Any]:
        """Match task complexity to best suited agent capability"""
        try:
            if task_id not in self.task_profiles:
                return {'recommended_agent': available_agents[0] if available_agents else None}
            
            task_profile = self.task_profiles[task_id]
            task_complexity = task_profile.adjusted_complexity
            
            # Evaluate each available agent
            agent_scores = {}
            
            for agent_id in available_agents:
                if agent_id in self.agent_capabilities:
                    agent_profile = self.agent_capabilities[agent_id]
                    
                    # Calculate capability match score
                    capability_diff = abs(agent_profile.current_capability - task_complexity)
                    capability_score = max(0, 1.0 - capability_diff * 2)  # Penalize large differences
                    
                    # Factor in optimal complexity range
                    optimal_range = agent_profile.optimal_complexity_range
                    in_optimal_range = optimal_range[0] <= task_complexity <= optimal_range[1]
                    range_bonus = 0.2 if in_optimal_range else 0.0
                    
                    # Factor in specialization
                    task_type = task_profile.complexity_factors.get('task_type', 'general')
                    specialization_score = agent_profile.specialization_areas.get(task_type, 0.5)
                    
                    # Factor in learning velocity (prefer agents that learn faster for challenging tasks)
                    learning_factor = agent_profile.learning_velocity if task_complexity > 0.7 else 0.0
                    
                    # Combine scores
                    total_score = (
                        capability_score * 0.4 +
                        range_bonus +
                        specialization_score * 0.3 +
                        learning_factor * 0.1
                    )
                    
                    agent_scores[agent_id] = {
                        'total_score': total_score,
                        'capability_match': capability_score,
                        'in_optimal_range': in_optimal_range,
                        'specialization': specialization_score,
                        'learning_velocity': agent_profile.learning_velocity
                    }
                else:
                    # Unknown agent - give default score
                    agent_scores[agent_id] = {'total_score': 0.5}
            
            # Select best agent
            if agent_scores:
                best_agent = max(agent_scores.keys(), key=lambda x: agent_scores[x]['total_score'])
                best_score = agent_scores[best_agent]['total_score']
                
                recommendation = {
                    'recommended_agent': best_agent,
                    'confidence': best_score,
                    'agent_scores': agent_scores,
                    'task_complexity': task_complexity,
                    'match_quality': 'excellent' if best_score > 0.8 else 'good' if best_score > 0.6 else 'acceptable'
                }
                
                logger.info(f"Task-agent matching: {best_agent} (score: {best_score:.2f})")
                return recommendation
            else:
                return {'recommended_agent': available_agents[0] if available_agents else None}
                
        except Exception as e:
            logger.error(f"Error matching task to agent capability: {e}")
            return {'recommended_agent': available_agents[0] if available_agents else None}

    async def generate_progressive_learning_path(self, agent_id: str) -> List[Dict[str, Any]]:
        """Generate progressive learning path for agent capability development"""
        try:
            if agent_id not in self.agent_capabilities:
                return []
            
            agent_profile = self.agent_capabilities[agent_id]
            current_capability = agent_profile.current_capability
            
            # Define learning progression steps
            learning_steps = []
            step_increment = 0.1  # Increase complexity by 10% each step
            max_steps = 5
            
            for step in range(max_steps):
                target_complexity = min(1.0, current_capability + (step + 1) * step_increment)
                
                learning_step = {
                    'step_number': step + 1,
                    'target_complexity': target_complexity,
                    'complexity_level': self._determine_complexity_level(target_complexity).value,
                    'recommended_tasks': self._generate_task_recommendations(target_complexity),
                    'learning_objectives': self._generate_learning_objectives(current_capability, target_complexity),
                    'success_criteria': {
                        'min_success_rate': 0.8,
                        'efficiency_threshold': 0.7,
                        'confidence_threshold': 0.75
                    },
                    'estimated_duration': f"{2 ** step}-{2 ** (step + 1)} tasks"
                }
                
                learning_steps.append(learning_step)
            
            return learning_steps
            
        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return []

    def _generate_task_recommendations(self, target_complexity: float) -> List[str]:
        """Generate task recommendations for target complexity"""
        complexity_level = self._determine_complexity_level(target_complexity)
        
        task_recommendations = {
            ComplexityLevel.TRIVIAL: ["Simple tap actions", "Basic navigation"],
            ComplexityLevel.SIMPLE: ["Settings access", "App launching"],
            ComplexityLevel.MODERATE: ["WiFi toggle", "Basic configuration"],
            ComplexityLevel.COMPLEX: ["Multi-step workflows", "Conditional actions"],
            ComplexityLevel.ADVANCED: ["Complex integrations", "Error recovery"],
            ComplexityLevel.EXPERT: ["System-level testing", "Advanced debugging"]
        }
        
        return task_recommendations.get(complexity_level, ["General tasks"])

    def _generate_learning_objectives(self, current: float, target: float) -> List[str]:
        """Generate learning objectives for capability progression"""
        improvement = target - current
        
        if improvement <= 0.1:
            return ["Maintain current performance", "Optimize efficiency"]
        elif improvement <= 0.2:
            return ["Handle slightly more complex tasks", "Improve error handling"]
        elif improvement <= 0.3:
            return ["Master multi-step workflows", "Develop specialized skills"]
        else:
            return ["Significant capability expansion", "Advanced problem solving"]

    def get_complexity_analytics(self) -> Dict[str, Any]:
        """Get comprehensive complexity management analytics"""
        try:
            # Task complexity distribution
            complexity_distribution = {}
            for profile in self.task_profiles.values():
                level = profile.complexity_level.value
                complexity_distribution[level] = complexity_distribution.get(level, 0) + 1
            
            # Agent capability distribution  
            capability_distribution = {}
            for profile in self.agent_capabilities.values():
                capability_range = f"{int(profile.current_capability * 10) / 10:.1f}"
                capability_distribution[capability_range] = capability_distribution.get(capability_range, 0) + 1
            
            # Recent adjustments analysis
            recent_adjustments = [adj for adj in self.complexity_adjustments 
                                if time.time() - adj['timestamp'] < 3600]  # Last hour
            
            adjustment_stats = {
                'total_recent': len(recent_adjustments),
                'increases': sum(1 for adj in recent_adjustments if adj['adjustment'] > 0),
                'decreases': sum(1 for adj in recent_adjustments if adj['adjustment'] < 0),
                'avg_adjustment': np.mean([adj['adjustment'] for adj in recent_adjustments]) if recent_adjustments else 0
            }
            
            return {
                'manager_metrics': self.manager_metrics.copy(),
                'complexity_distribution': complexity_distribution,
                'capability_distribution': capability_distribution,
                'adjustment_statistics': adjustment_stats,
                'total_tasks_managed': len(self.task_profiles),
                'total_agents_tracked': len(self.agent_capabilities),
                'average_complexity': np.mean([p.adjusted_complexity for p in self.task_profiles.values()]) if self.task_profiles else 0,
                'average_capability': np.mean([p.current_capability for p in self.agent_capabilities.values()]) if self.agent_capabilities else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating complexity analytics: {e}")
            return {'error': str(e)}


__all__ = [
    "DynamicComplexityManager",
    "TaskComplexityProfile",
    "AgentCapabilityProfile", 
    "ComplexityLevel"
]