"""
Utility functions for Attention Economics calculations
Helper functions for attention-related computations and optimizations
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import statistics


@dataclass
class AttentionCostFactors:
    """Factors that influence attention cost calculations"""
    base_complexity: float = 1.0
    task_novelty: float = 1.0
    context_switching_penalty: float = 0.0
    collaboration_overhead: float = 0.0
    learning_bonus: float = 0.0
    urgency_multiplier: float = 1.0
    confidence_modifier: float = 1.0


def calculate_base_attention_cost(task_description: str, context: Dict[str, Any] = None) -> float:
    """
    Calculate base attention cost for a task
    
    Args:
        task_description: Description of the task
        context: Additional context information
        
    Returns:
        Base attention cost (0.1 to 10.0)
    """
    try:
        # Base cost factors
        base_cost = 1.0
        
        # Task complexity based on description length and keywords
        word_count = len(task_description.split())
        complexity_factor = min(3.0, 1.0 + (word_count / 10))
        
        # Complexity keywords
        complex_keywords = [
            'analyze', 'complex', 'comprehensive', 'multi-step',
            'coordinate', 'integrate', 'optimize', 'evaluate'
        ]
        
        simple_keywords = [
            'click', 'tap', 'open', 'close', 'simple', 'basic'
        ]
        
        task_lower = task_description.lower()
        
        # Increase cost for complex keywords
        complex_matches = sum(1 for keyword in complex_keywords if keyword in task_lower)
        complexity_factor += complex_matches * 0.3
        
        # Decrease cost for simple keywords
        simple_matches = sum(1 for keyword in simple_keywords if keyword in task_lower)
        complexity_factor = max(0.3, complexity_factor - simple_matches * 0.2)
        
        # Context-based adjustments
        if context:
            # Time pressure increases cost
            if context.get('urgent', False):
                complexity_factor *= 1.4
            
            # Previous failures increase cost
            if context.get('retry_count', 0) > 0:
                complexity_factor *= (1.0 + context['retry_count'] * 0.2)
            
            # Task familiarity reduces cost
            familiarity = context.get('familiarity_score', 0.5)
            complexity_factor *= max(0.5, 1.0 - (familiarity - 0.5))
        
        final_cost = base_cost * complexity_factor
        
        # Clamp to reasonable range
        return max(0.1, min(10.0, final_cost))
        
    except Exception:
        return 2.0  # Default cost


def calculate_planning_attention_cost(plan_complexity: str, steps_count: int, 
                                    risk_factors: List[str] = None) -> float:
    """
    Calculate attention cost for planning activities
    
    Args:
        plan_complexity: Complexity level ('simple', 'moderate', 'complex', 'highly_complex')
        steps_count: Number of steps in the plan
        risk_factors: List of identified risk factors
        
    Returns:
        Planning attention cost
    """
    try:
        # Base costs by complexity
        complexity_costs = {
            'simple': 1.0,
            'moderate': 2.0,
            'complex': 3.5,
            'highly_complex': 5.0
        }
        
        base_cost = complexity_costs.get(plan_complexity, 2.0)
        
        # Steps factor
        steps_factor = 1.0 + (steps_count * 0.1)
        
        # Risk factor penalty
        risk_penalty = 1.0
        if risk_factors:
            high_risk_factors = [
                'destructive_operation_risk',
                'system_modification_risk',
                'single_attempt_risk'
            ]
            
            high_risk_count = sum(1 for risk in risk_factors if risk in high_risk_factors)
            risk_penalty = 1.0 + (len(risk_factors) * 0.1) + (high_risk_count * 0.2)
        
        final_cost = base_cost * steps_factor * risk_penalty
        
        return max(1.0, min(15.0, final_cost))
        
    except Exception:
        return 3.0


def calculate_execution_attention_cost(action_type: str, execution_context: Dict[str, Any] = None) -> float:
    """
    Calculate attention cost for execution activities
    
    Args:
        action_type: Type of action being executed
        execution_context: Context information for execution
        
    Returns:
        Execution attention cost
    """
    try:
        # Base costs by action type
        action_costs = {
            'ui_interaction': 1.0,
            'navigation': 0.8,
            'text_input': 1.2,
            'verification': 1.5,
            'analysis': 2.0,
            'coordination': 2.5,
            'error_handling': 3.0,
            'adaptation': 2.2
        }
        
        base_cost = action_costs.get(action_type, 1.5)
        
        # Context adjustments
        if execution_context:
            # Device compatibility affects cost
            compatibility_score = execution_context.get('device_compatibility', 1.0)
            if compatibility_score < 0.8:
                base_cost *= 1.3
            
            # Previous attempt failures
            attempt_number = execution_context.get('attempt_number', 1)
            if attempt_number > 1:
                base_cost *= (1.0 + (attempt_number - 1) * 0.15)
            
            # Uncertainty increases cost
            confidence = execution_context.get('confidence', 0.8)
            uncertainty_penalty = max(1.0, 1.5 - confidence)
            base_cost *= uncertainty_penalty
        
        return max(0.5, min(8.0, base_cost))
        
    except Exception:
        return 1.5


def calculate_supervision_attention_cost(agents_count: int, coordination_complexity: str,
                                       monitoring_level: str = 'standard') -> float:
    """
    Calculate attention cost for supervision activities
    
    Args:
        agents_count: Number of agents being supervised
        coordination_complexity: Complexity of coordination ('simple', 'moderate', 'complex')
        monitoring_level: Level of monitoring ('low', 'standard', 'high')
        
    Returns:
        Supervision attention cost
    """
    try:
        # Base cost for supervision
        base_cost = 2.0
        
        # Agent count factor
        agent_factor = 1.0 + (agents_count * 0.3)
        
        # Coordination complexity factor
        complexity_factors = {
            'simple': 1.0,
            'moderate': 1.4,
            'complex': 2.0
        }
        complexity_factor = complexity_factors.get(coordination_complexity, 1.4)
        
        # Monitoring level factor
        monitoring_factors = {
            'low': 0.8,
            'standard': 1.0,
            'high': 1.3
        }
        monitoring_factor = monitoring_factors.get(monitoring_level, 1.0)
        
        final_cost = base_cost * agent_factor * complexity_factor * monitoring_factor
        
        return max(1.0, min(20.0, final_cost))
        
    except Exception:
        return 3.0


def calculate_learning_attention_cost(learning_type: str, pattern_complexity: float = 0.5) -> float:
    """
    Calculate attention cost for learning activities
    
    Args:
        learning_type: Type of learning ('pattern_recognition', 'adaptation', 'generalization')
        pattern_complexity: Complexity of the pattern being learned (0.0 to 1.0)
        
    Returns:
        Learning attention cost
    """
    try:
        # Base costs by learning type
        learning_costs = {
            'pattern_recognition': 1.5,
            'adaptation': 2.0,
            'generalization': 2.5,
            'meta_learning': 3.0
        }
        
        base_cost = learning_costs.get(learning_type, 2.0)
        
        # Complexity factor
        complexity_factor = 1.0 + pattern_complexity
        
        final_cost = base_cost * complexity_factor
        
        return max(0.8, min(6.0, final_cost))
        
    except Exception:
        return 2.0


def optimize_attention_allocation(available_attention: float, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optimize attention allocation across multiple tasks
    
    Args:
        available_attention: Total available attention units
        tasks: List of tasks with attention requirements and priorities
        
    Returns:
        Optimized task allocation with attention assignments
    """
    try:
        if not tasks or available_attention <= 0:
            return []
        
        # Sort tasks by priority and efficiency
        def task_efficiency(task):
            attention_cost = task.get('attention_cost', 1.0)
            priority = task.get('priority', 1.0)
            expected_value = task.get('expected_value', 1.0)
            
            # Efficiency = value per attention unit
            return (priority * expected_value) / max(0.1, attention_cost)
        
        sorted_tasks = sorted(tasks, key=task_efficiency, reverse=True)
        
        # Allocate attention using greedy approach
        allocated_tasks = []
        remaining_attention = available_attention
        
        for task in sorted_tasks:
            required_attention = task.get('attention_cost', 1.0)
            
            if remaining_attention >= required_attention:
                # Full allocation
                allocated_task = task.copy()
                allocated_task['allocated_attention'] = required_attention
                allocated_task['allocation_ratio'] = 1.0
                allocated_tasks.append(allocated_task)
                remaining_attention -= required_attention
            
            elif remaining_attention > 0.1:
                # Partial allocation if task supports it
                if task.get('supports_partial', False):
                    allocated_task = task.copy()
                    allocated_task['allocated_attention'] = remaining_attention
                    allocated_task['allocation_ratio'] = remaining_attention / required_attention
                    allocated_tasks.append(allocated_task)
                    remaining_attention = 0
                    break
        
        return allocated_tasks
        
    except Exception:
        return []


def calculate_attention_waste(allocated_attention: float, actual_usage: float, 
                            task_success: bool) -> float:
    """
    Calculate attention waste for a completed task
    
    Args:
        allocated_attention: Amount of attention allocated
        actual_usage: Amount of attention actually used
        task_success: Whether the task was successful
        
    Returns:
        Attention waste ratio (0.0 to 1.0)
    """
    try:
        if allocated_attention <= 0:
            return 0.0
        
        # Basic waste calculation
        if actual_usage <= allocated_attention:
            # Under-utilization
            waste_ratio = (allocated_attention - actual_usage) / allocated_attention
        else:
            # Over-utilization (also a form of waste)
            waste_ratio = (actual_usage - allocated_attention) / allocated_attention
        
        # Penalty for failed tasks
        if not task_success:
            waste_ratio = min(1.0, waste_ratio + 0.3)
        
        return max(0.0, min(1.0, waste_ratio))
        
    except Exception:
        return 0.5


def calculate_collaboration_bonus(agent_count: int, coordination_success: bool,
                                shared_knowledge: float = 0.0) -> float:
    """
    Calculate collaboration bonus for multi-agent tasks
    
    Args:
        agent_count: Number of agents collaborating
        coordination_success: Whether coordination was successful
        shared_knowledge: Amount of knowledge shared (0.0 to 1.0)
        
    Returns:
        Collaboration bonus multiplier (0.8 to 1.5)
    """
    try:
        if agent_count <= 1:
            return 1.0
        
        # Base bonus for collaboration
        base_bonus = 1.0
        
        if coordination_success:
            # Positive collaboration effects
            collaboration_factor = min(1.4, 1.0 + (agent_count - 1) * 0.1)
            knowledge_bonus = shared_knowledge * 0.2
            base_bonus = collaboration_factor + knowledge_bonus
        else:
            # Coordination failure penalty
            penalty = min(0.3, (agent_count - 1) * 0.05)
            base_bonus = max(0.7, 1.0 - penalty)
        
        return max(0.8, min(1.5, base_bonus))
        
    except Exception:
        return 1.0


def calculate_adaptive_attention_rate(current_load: float, performance_history: List[float],
                                    target_efficiency: float = 0.8) -> float:
    """
    Calculate adaptive attention recharge rate based on current system state
    
    Args:
        current_load: Current attention load (0.0 to 1.0)
        performance_history: Recent performance scores
        target_efficiency: Target efficiency level
        
    Returns:
        Adaptive recharge rate multiplier
    """
    try:
        base_rate = 1.0
        
        # Load-based adjustment
        if current_load > 0.8:
            # High load - increase recharge rate
            load_bonus = (current_load - 0.8) * 2.0  # Up to 40% bonus
            base_rate += load_bonus
        elif current_load < 0.3:
            # Low load - decrease recharge rate to save resources
            load_penalty = (0.3 - current_load) * 1.0  # Up to 30% reduction
            base_rate -= load_penalty
        
        # Performance-based adjustment
        if performance_history:
            recent_performance = statistics.mean(performance_history[-5:])  # Last 5 measurements
            
            if recent_performance < target_efficiency:
                # Poor performance - increase recharge rate
                performance_bonus = (target_efficiency - recent_performance) * 1.5
                base_rate += performance_bonus
        
        return max(0.5, min(2.0, base_rate))
        
    except Exception:
        return 1.0


def estimate_task_attention_requirement(task_type: str, context: Dict[str, Any] = None) -> Tuple[float, float]:
    """
    Estimate attention requirement for a task with confidence bounds
    
    Args:
        task_type: Type of task
        context: Additional context information
        
    Returns:
        Tuple of (estimated_cost, confidence_interval)
    """
    try:
        # Base estimates by task type
        base_estimates = {
            'simple_interaction': (1.0, 0.2),
            'complex_analysis': (4.0, 1.0),
            'multi_step_process': (3.0, 0.8),
            'learning_task': (2.5, 0.6),
            'coordination_task': (3.5, 1.2),
            'error_recovery': (2.8, 0.9)
        }
        
        base_cost, base_confidence = base_estimates.get(task_type, (2.0, 0.5))
        
        # Context adjustments
        if context:
            # Complexity adjustment
            complexity = context.get('complexity_score', 0.5)
            complexity_multiplier = 0.5 + complexity
            base_cost *= complexity_multiplier
            
            # Uncertainty increases confidence interval
            uncertainty = context.get('uncertainty', 0.3)
            base_confidence *= (1.0 + uncertainty)
            
            # Historical data improves estimate
            if context.get('historical_data'):
                historical_costs = context['historical_data']
                if len(historical_costs) >= 3:
                    historical_mean = statistics.mean(historical_costs)
                    historical_std = statistics.stdev(historical_costs)
                    
                    # Blend with historical data
                    base_cost = (base_cost + historical_mean) / 2
                    base_confidence = min(base_confidence, historical_std)
        
        return (max(0.1, base_cost), max(0.1, base_confidence))
        
    except Exception:
        return (2.0, 0.5)


def calculate_attention_recovery_time(depletion_level: float, recovery_rate: float = 1.0) -> float:
    """
    Calculate time needed to recover attention to full capacity
    
    Args:
        depletion_level: Current depletion level (0.0 to 1.0)
        recovery_rate: Recovery rate multiplier
        
    Returns:
        Recovery time in seconds
    """
    try:
        if depletion_level <= 0:
            return 0.0
        
        # Base recovery function (exponential recovery)
        base_recovery_time = -math.log(1 - depletion_level) * 60  # Base: 60 seconds for full recovery
        
        # Apply recovery rate
        actual_recovery_time = base_recovery_time / max(0.1, recovery_rate)
        
        return max(1.0, actual_recovery_time)
        
    except Exception:
        return 60.0


def optimize_attention_scheduling(tasks: List[Dict[str, Any]], 
                                time_horizon: float = 3600.0) -> List[Dict[str, Any]]:
    """
    Optimize attention scheduling over time
    
    Args:
        tasks: List of tasks with timing and attention requirements
        time_horizon: Planning horizon in seconds
        
    Returns:
        Optimized schedule with timing and attention allocation
    """
    try:
        if not tasks:
            return []
        
        # Sort by deadline and priority
        def scheduling_priority(task):
            deadline = task.get('deadline', time_horizon)
            priority = task.get('priority', 1.0)
            attention_cost = task.get('attention_cost', 1.0)
            
            # Urgency factor
            urgency = max(0.1, time_horizon - deadline) / time_horizon
            
            return priority / (urgency * attention_cost)
        
        sorted_tasks = sorted(tasks, key=scheduling_priority, reverse=True)
        
        # Schedule tasks with attention recovery
        schedule = []
        current_time = 0.0
        current_attention = 10.0  # Assume full attention capacity
        
        for task in sorted_tasks:
            required_attention = task.get('attention_cost', 1.0)
            task_duration = task.get('estimated_duration', 30.0)
            
            # Check if we have enough attention
            if current_attention < required_attention:
                # Calculate recovery time needed
                attention_deficit = required_attention - current_attention
                recovery_time = calculate_attention_recovery_time(attention_deficit / 10.0)
                
                # Add recovery period
                current_time += recovery_time
                current_attention = min(10.0, current_attention + (recovery_time / 60.0))
            
            # Schedule the task
            if current_attention >= required_attention:
                scheduled_task = task.copy()
                scheduled_task['scheduled_start'] = current_time
                scheduled_task['scheduled_end'] = current_time + task_duration
                scheduled_task['attention_allocated'] = required_attention
                
                schedule.append(scheduled_task)
                
                # Update state
                current_time += task_duration
                current_attention -= required_attention
                
                # Natural attention recovery during task
                recovery_during_task = (task_duration / 60.0) * 0.3  # Partial recovery
                current_attention = min(10.0, current_attention + recovery_during_task)
        
        return schedule
        
    except Exception:
        return []


__all__ = [
    "AttentionCostFactors",
    "calculate_base_attention_cost",
    "calculate_planning_attention_cost", 
    "calculate_execution_attention_cost",
    "calculate_supervision_attention_cost",
    "calculate_learning_attention_cost",
    "optimize_attention_allocation",
    "calculate_attention_waste",
    "calculate_collaboration_bonus",
    "calculate_adaptive_attention_rate",
    "estimate_task_attention_requirement",
    "calculate_attention_recovery_time",
    "optimize_attention_scheduling"
]