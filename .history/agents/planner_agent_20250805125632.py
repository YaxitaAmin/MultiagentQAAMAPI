"""
Planner Agent for AMAPI System
Specializes in task planning, strategy development, and goal decomposition
"""

import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from loguru import logger

from agents.base_agent import BaseQAAgent, AgentType
from core.attention_economics import AttentionEconomicsEngine, AttentionAllocation
from core.behavioral_learning import BehavioralPatternEngine, LearningType
from core.llm_interface import LLMInterface
from utils.attention_helpers import calculate_planning_attention_cost


class PlanningStrategy(Enum):
    """Planning strategies for different scenarios"""
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    INCREMENTAL = "incremental"


class PlanComplexity(Enum):
    """Plan complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class QAPlanStep:
    """Individual step in a QA plan"""
    step_id: str
    step_number: int
    action: str
    description: str
    prerequisites: List[str]
    expected_outcome: str
    validation_criteria: List[str]
    estimated_duration: float
    risk_level: str
    fallback_options: List[str]
    attention_cost: float = 1.0
    confidence: float = 0.8


@dataclass 
class QAPlan:
    """Comprehensive QA execution plan"""
    plan_id: str
    goal: str
    description: str
    strategy: PlanningStrategy
    complexity: PlanComplexity
    execution_steps: List[QAPlanStep]
    success_criteria: List[str]
    risk_assessment: Dict[str, Any]
    resource_requirements: Dict[str, float]
    estimated_total_time: float
    confidence_score: float
    created_timestamp: float
    created_by: str
    dependencies: List[str] = None
    alternatives: List[str] = None
    monitoring_points: List[str] = None


@dataclass
class PlanningContext:
    """Context information for planning"""
    task_type: str
    domain: str
    constraints: List[str]
    available_resources: Dict[str, Any]
    time_constraints: Optional[float]
    quality_requirements: List[str]
    risk_tolerance: str
    previous_attempts: List[str] = None


@dataclass
class PlanAnalysis:
    """Analysis of a planning request"""
    complexity_score: float
    required_capabilities: List[str]
    risk_factors: List[str]
    resource_demands: Dict[str, float]
    success_probability: float
    alternative_approaches: List[str]
    optimization_opportunities: List[str]


class PlannerAgent(BaseQAAgent):
    """
    Advanced Planner Agent with AMAPI Integration
    Specializes in intelligent task planning and strategy development
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            agent_type=AgentType.PLANNER,
            config=config
        )
        
        # Planning components
        self.llm_interface = LLMInterface(config.get('llm', {}))
        self.behavioral_engine = BehavioralPatternEngine(config.get('behavioral', {}))
        
        # Plan storage and management
        self.active_plans: Dict[str, QAPlan] = {}
        self.plan_templates: Dict[str, Dict[str, Any]] = {}
        self.planning_history: List[QAPlan] = []
        
        # Planning knowledge base
        self.domain_knowledge: Dict[str, List[str]] = {}
        self.strategy_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.risk_assessments: Dict[str, Dict[str, float]] = {}
        
        # Planner specializations
        self.specializations = [
            'task_decomposition',
            'strategy_development', 
            'risk_assessment',
            'resource_planning',
            'goal_optimization',
            'contingency_planning'
        ]
        
        # Planning metrics
        self.planning_metrics = {
            'plans_created': 0,
            'successful_plans': 0,
            'average_planning_time': 0.0,
            'complexity_handled': 0.0,
            'risk_prediction_accuracy': 0.0,
            'plan_optimization_rate': 0.0,
            'template_reuse_rate': 0.0,
            'adaptive_replanning_events': 0
        }
        
        self.logger.info("Planner Agent initialized")

    async def _initialize_agent_systems(self) -> None:
        """Initialize planner-specific systems"""
        try:
            # Initialize LLM interface
            await self.llm_interface.initialize()
            
            # Start behavioral learning engine
            await self.behavioral_engine.start_learning_engine()
            
            # Load planning templates and patterns
            await self._load_planning_templates()
            await self._load_domain_knowledge()
            await self._load_strategy_patterns()
            
            self.logger.info("Planner agent systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize planner systems: {e}")
            raise

    async def _cleanup_agent_systems(self) -> None:
        """Cleanup planner-specific systems"""
        try:
            await self.llm_interface.cleanup()
            await self.behavioral_engine.stop_learning_engine()
            await self._save_planning_knowledge()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up planner systems: {e}")

    async def predict(self, instruction: str, observation: Dict[str, Any], 
                     context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Create comprehensive QA plan with AMAPI enhancements"""
        try:
            # Allocate attention for planning
            attention_allocation = await self.allocate_attention(
                f"Plan: {instruction}",
                context.get('task_complexity', 0.7) if context else 0.7
            )
            
            # Analyze planning requirements
            planning_analysis = await self._analyze_planning_requirements(
                instruction, observation, context
            )
            
            # Create comprehensive plan
            qa_plan = await self._create_comprehensive_plan(
                instruction, planning_analysis, attention_allocation, context
            )
            
            # Optimize plan
            optimized_plan = await self._optimize_plan(qa_plan, planning_analysis)
            
            # Store plan
            self.active_plans[optimized_plan.plan_id] = optimized_plan
            
            # Create planning actions
            actions = await self._create_planning_actions(optimized_plan)
            
            # Update planning metrics
            await self._update_planning_metrics(optimized_plan, attention_allocation)
            
            # Record learning event
            await self.behavioral_engine.record_learning_event(
                self.agent_id,
                LearningType.SUCCESS_PATTERN,
                {
                    'planning_strategy': optimized_plan.strategy.value,
                    'complexity': optimized_plan.complexity.value,
                    'steps_count': len(optimized_plan.execution_steps),
                    'confidence': optimized_plan.confidence_score
                },
                'success',
                {'confidence': optimized_plan.confidence_score}
            )
            
            # Create reasoning info
            reasoning_info = {
                'plan_created': True,
                'plan_id': optimized_plan.plan_id,
                'planning_strategy': optimized_plan.strategy.value,
                'plan_complexity': optimized_plan.complexity.value,
                'steps_count': len(optimized_plan.execution_steps),
                'estimated_duration': optimized_plan.estimated_total_time,
                'confidence_score': optimized_plan.confidence_score,
                'risk_assessment': optimized_plan.risk_assessment,
                'resource_requirements': optimized_plan.resource_requirements,
                'attention_efficiency': attention_allocation.efficiency_score,
                'planning_analysis': asdict(planning_analysis),
                'success_criteria': optimized_plan.success_criteria,
                'monitoring_points': optimized_plan.monitoring_points or []
            }
            
            self.logger.info(f"Plan created successfully: {optimized_plan.plan_id}")
            
            return reasoning_info, actions
            
        except Exception as e:
            self.logger.error(f"Error in planner prediction: {e}")
            return {'error': str(e), 'confidence': 0.0}, []

    async def _analyze_planning_requirements(self, instruction: str, observation: Dict[str, Any],
                                           context: Dict[str, Any] = None) -> PlanAnalysis:
        """Analyze planning requirements and constraints"""
        try:
            # Extract planning context
            planning_context = self._extract_planning_context(instruction, context)
            
            # Analyze task complexity
            complexity_score = await self._assess_task_complexity(instruction, planning_context)
            
            # Identify required capabilities
            required_capabilities = await self._identify_required_capabilities(instruction, planning_context)
            
            # Assess risk factors
            risk_factors = await self._assess_risk_factors(instruction, planning_context)
            
            # Calculate resource demands
            resource_demands = await self._calculate_resource_demands(
                instruction, complexity_score, required_capabilities
            )
            
            # Estimate success probability
            success_probability = await self._estimate_success_probability(
                complexity_score, required_capabilities, risk_factors
            )
            
            # Generate alternative approaches
            alternative_approaches = await self._generate_alternative_approaches(
                instruction, planning_context
            )
            
            # Identify optimization opportunities
            optimization_opportunities = await self._identify_optimization_opportunities(
                instruction, planning_context, complexity_score
            )
            
            analysis = PlanAnalysis(
                complexity_score=complexity_score,
                required_capabilities=required_capabilities,
                risk_factors=risk_factors,
                resource_demands=resource_demands,
                success_probability=success_probability,
                alternative_approaches=alternative_approaches,
                optimization_opportunities=optimization_opportunities
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing planning requirements: {e}")
            return PlanAnalysis(
                complexity_score=0.5,
                required_capabilities=[],
                risk_factors=[],
                resource_demands={},
                success_probability=0.7,
                alternative_approaches=[],
                optimization_opportunities=[]
            )

    def _extract_planning_context(self, instruction: str, context: Dict[str, Any] = None) -> PlanningContext:
        """Extract planning context from instruction and context"""
        try:
            # Classify task type
            task_type = self._classify_task_type(instruction)
            
            # Determine domain
            domain = self._determine_domain(instruction)
            
            # Extract constraints
            constraints = self._extract_constraints(instruction, context)
            
            # Get available resources
            available_resources = context.get('available_resources', {}) if context else {}
            
            # Extract time constraints
            time_constraints = context.get('time_limit') if context else None
            
            # Determine quality requirements
            quality_requirements = self._extract_quality_requirements(instruction)
            
            # Assess risk tolerance
            risk_tolerance = context.get('risk_tolerance', 'medium') if context else 'medium'
            
            return PlanningContext(
                task_type=task_type,
                domain=domain,
                constraints=constraints,
                available_resources=available_resources,
                time_constraints=time_constraints,
                quality_requirements=quality_requirements,
                risk_tolerance=risk_tolerance,
                previous_attempts=context.get('previous_attempts', []) if context else []
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting planning context: {e}")
            return PlanningContext(
                task_type='general',
                domain='unknown',
                constraints=[],
                available_resources={},
                time_constraints=None,
                quality_requirements=[],
                risk_tolerance='medium'
            )

    def _classify_task_type(self, instruction: str) -> str:
        """Classify the type of task for planning"""
        instruction_lower = instruction.lower()
        
        if any(word in instruction_lower for word in ['test', 'verify', 'check', 'validate']):
            return 'testing'
        elif any(word in instruction_lower for word in ['automate', 'execute', 'perform', 'run']):
            return 'automation'
        elif any(word in instruction_lower for word in ['analyze', 'examine', 'investigate', 'study']):
            return 'analysis'
        elif any(word in instruction_lower for word in ['setup', 'configure', 'install', 'deploy']):
            return 'configuration'
        elif any(word in instruction_lower for word in ['navigate', 'browse', 'search', 'find']):
            return 'navigation'
        elif any(word in instruction_lower for word in ['data', 'information', 'extract', 'collect']):
            return 'data_processing'
        else:
            return 'general'

    def _determine_domain(self, instruction: str) -> str:
        """Determine the domain of the task"""
        instruction_lower = instruction.lower()
        
        if any(word in instruction_lower for word in ['app', 'application', 'mobile', 'android']):
            return 'mobile_app'
        elif any(word in instruction_lower for word in ['web', 'website', 'browser', 'internet']):
            return 'web'
        elif any(word in instruction_lower for word in ['system', 'settings', 'configuration']):
            return 'system'
        elif any(word in instruction_lower for word in ['file', 'document', 'folder', 'storage']):
            return 'file_system'
        elif any(word in instruction_lower for word in ['network', 'wifi', 'connection', 'internet']):
            return 'network'
        else:
            return 'general'

    def _extract_constraints(self, instruction: str, context: Dict[str, Any] = None) -> List[str]:
        """Extract constraints from instruction and context"""
        constraints = []
        
        instruction_lower = instruction.lower()
        
        # Time constraints
        if any(word in instruction_lower for word in ['quickly', 'fast', 'urgent', 'immediate']):
            constraints.append('time_critical')
        
        # Quality constraints
        if any(word in instruction_lower for word in ['carefully', 'accurate', 'precise', 'thorough']):
            constraints.append('high_quality_required')
        
        # Resource constraints
        if context:
            if context.get('limited_resources'):
                constraints.append('resource_limited')
            if context.get('single_attempt'):
                constraints.append('no_retry_allowed')
        
        return constraints

    def _extract_quality_requirements(self, instruction: str) -> List[str]:
        """Extract quality requirements from instruction"""
        requirements = []
        instruction_lower = instruction.lower()
        
        if 'accurate' in instruction_lower or 'precise' in instruction_lower:
            requirements.append('high_accuracy')
        if 'thorough' in instruction_lower or 'complete' in instruction_lower:
            requirements.append('completeness')
        if 'reliable' in instruction_lower or 'consistent' in instruction_lower:
            requirements.append('reliability')
        if 'fast' in instruction_lower or 'efficient' in instruction_lower:
            requirements.append('efficiency')
        
        return requirements

    async def _assess_task_complexity(self, instruction: str, planning_context: PlanningContext) -> float:
        """Assess the complexity of the task"""
        try:
            complexity_factors = []
            
            # Base complexity from instruction length and keywords
            word_count = len(instruction.split())
            complexity_factors.append(min(1.0, word_count / 20))  # Normalize to 0-1
            
            # Complexity from task type
            task_complexity_map = {
                'testing': 0.7,
                'automation': 0.8,
                'analysis': 0.6,
                'configuration': 0.7,
                'navigation': 0.4,
                'data_processing': 0.8,
                'general': 0.5
            }
            complexity_factors.append(task_complexity_map.get(planning_context.task_type, 0.5))
            
            # Complexity from constraints
            constraint_complexity = len(planning_context.constraints) * 0.1
            complexity_factors.append(min(1.0, constraint_complexity))
            
            # Complexity from quality requirements
            quality_complexity = len(planning_context.quality_requirements) * 0.15
            complexity_factors.append(min(1.0, quality_complexity))
            
            # Complexity from domain
            domain_complexity_map = {
                'mobile_app': 0.8,
                'web': 0.7,
                'system': 0.9,
                'file_system': 0.5,
                'network': 0.8,
                'general': 0.5
            }
            complexity_factors.append(domain_complexity_map.get(planning_context.domain, 0.5))
            
            # Calculate weighted average
            weights = [0.2, 0.3, 0.2, 0.15, 0.15]
            complexity_score = sum(f * w for f, w in zip(complexity_factors, weights))
            
            return min(1.0, max(0.1, complexity_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing task complexity: {e}")
            return 0.5

    async def _identify_required_capabilities(self, instruction: str, 
                                            planning_context: PlanningContext) -> List[str]:
        """Identify capabilities required for the task"""
        try:
            capabilities = set()
            
            # Capabilities from task type
            task_capability_map = {
                'testing': ['ui_automation', 'verification', 'assertion_checking'],
                'automation': ['ui_automation', 'workflow_execution', 'error_handling'],
                'analysis': ['data_analysis', 'pattern_recognition', 'reporting'],
                'configuration': ['system_configuration', 'settings_management'],
                'navigation': ['ui_navigation', 'element_finding', 'path_planning'],
                'data_processing': ['data_extraction', 'data_transformation', 'data_validation']
            }
            
            task_capabilities = task_capability_map.get(planning_context.task_type, [])
            capabilities.update(task_capabilities)
            
            # Capabilities from domain
            domain_capability_map = {
                'mobile_app': ['android_automation', 'touch_gestures', 'app_navigation'],
                'web': ['web_automation', 'browser_control', 'web_element_handling'],
                'system': ['system_control', 'settings_access', 'permission_management'],
                'file_system': ['file_operations', 'directory_navigation', 'file_validation'],
                'network': ['network_testing', 'connectivity_checking', 'protocol_handling']
            }
            
            domain_capabilities = domain_capability_map.get(planning_context.domain, [])
            capabilities.update(domain_capabilities)
            
            # Capabilities from instruction keywords
            instruction_lower = instruction.lower()
            if 'screenshot' in instruction_lower or 'capture' in instruction_lower:
                capabilities.add('screenshot_capture')
            if 'input' in instruction_lower or 'type' in instruction_lower:
                capabilities.add('text_input')
            if 'click' in instruction_lower or 'tap' in instruction_lower:
                capabilities.add('ui_interaction')
            if 'wait' in instruction_lower or 'delay' in instruction_lower:
                capabilities.add('timing_control')
            if 'scroll' in instruction_lower or 'swipe' in instruction_lower:
                capabilities.add('gesture_control')
            
            return list(capabilities)
            
        except Exception as e:
            self.logger.error(f"Error identifying required capabilities: {e}")
            return ['general_automation']

    async def _assess_risk_factors(self, instruction: str, planning_context: PlanningContext) -> List[str]:
        """Assess risk factors for the task"""
        try:
            risk_factors = []
            
            # Complexity-based risks
            if planning_context.task_type in ['automation', 'system', 'data_processing']:
                risk_factors.append('high_complexity_execution')
            
            # Domain-based risks
            if planning_context.domain in ['system', 'network']:
                risk_factors.append('system_modification_risk')
            
            # Constraint-based risks
            if 'time_critical' in planning_context.constraints:
                risk_factors.append('time_pressure_risk')
            if 'no_retry_allowed' in planning_context.constraints:
                risk_factors.append('single_attempt_risk')
            
            # Instruction-based risks
            instruction_lower = instruction.lower()
            if any(word in instruction_lower for word in ['delete', 'remove', 'uninstall']):
                risk_factors.append('destructive_operation_risk')
            if any(word in instruction_lower for word in ['permission', 'access', 'admin']):
                risk_factors.append('permission_elevation_risk')
            if any(word in instruction_lower for word in ['network', 'connection', 'internet']):
                risk_factors.append('network_dependency_risk')
            
            # Resource-based risks
            if not planning_context.available_resources:
                risk_factors.append('resource_uncertainty_risk')
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Error assessing risk factors: {e}")
            return ['general_execution_risk']

    async def _calculate_resource_demands(self, instruction: str, complexity_score: float,
                                        required_capabilities: List[str]) -> Dict[str, float]:
        """Calculate resource demands for the task"""
        try:
            demands = {}
            
            # Attention demand based on complexity
            demands['attention'] = 2.0 + (complexity_score * 3.0)
            
            # Time demand based on complexity and capabilities
            base_time = 30.0  # seconds
            complexity_multiplier = 1.0 + complexity_score
            capability_multiplier = 1.0 + (len(required_capabilities) * 0.1)
            demands['time'] = base_time * complexity_multiplier * capability_multiplier
            
            # Memory demand for storing intermediate results
            demands['memory'] = 1.0 + (complexity_score * 0.5)
            
            # Processing power for complex operations
            demands['processing'] = complexity_score * 2.0
            
            # Network resources if needed
            if any('network' in cap or 'web' in cap for cap in required_capabilities):
                demands['network'] = 1.0
            
            return demands
            
        except Exception as e:
            self.logger.error(f"Error calculating resource demands: {e}")
            return {'attention': 2.0, 'time': 30.0, 'memory': 1.0}

    async def _estimate_success_probability(self, complexity_score: float, 
                                          required_capabilities: List[str],
                                          risk_factors: List[str]) -> float:
        """Estimate probability of successful execution"""
        try:
            base_success_rate = 0.8
            
            # Reduce success rate based on complexity
            complexity_penalty = complexity_score * 0.3
            
            # Reduce success rate based on number of required capabilities
            capability_penalty = len(required_capabilities) * 0.02
            
            # Reduce success rate based on risk factors
            risk_penalty = len(risk_factors) * 0.05
            
            # High-risk operations have additional penalty
            high_risk_operations = [
                'destructive_operation_risk',
                'system_modification_risk',
                'permission_elevation_risk'
            ]
            
            if any(risk in high_risk_operations for risk in risk_factors):
                risk_penalty += 0.1
            
            # Calculate final success probability
            success_probability = base_success_rate - complexity_penalty - capability_penalty - risk_penalty
            
            return max(0.1, min(0.95, success_probability))
            
        except Exception as e:
            self.logger.error(f"Error estimating success probability: {e}")
            return 0.7

    async def _generate_alternative_approaches(self, instruction: str, 
                                             planning_context: PlanningContext) -> List[str]:
        """Generate alternative approaches for the task"""
        try:
            alternatives = []
            
            # Strategy-based alternatives
            if planning_context.task_type == 'automation':
                alternatives.extend([
                    'step_by_step_manual_execution',
                    'batch_operation_approach',
                    'template_based_execution'
                ])
            
            elif planning_context.task_type == 'testing':
                alternatives.extend([
                    'exploratory_testing_approach',
                    'checklist_based_testing',
                    'automated_regression_testing'
                ])
            
            elif planning_context.task_type == 'analysis':
                alternatives.extend([
                    'statistical_analysis_approach',
                    'pattern_matching_approach',
                    'comparative_analysis_approach'
                ])
            
            # Risk-based alternatives
            if 'high_complexity_execution' in planning_context.constraints:
                alternatives.append('simplified_approach_with_manual_verification')
            
            if 'time_critical' in planning_context.constraints:
                alternatives.append('parallel_execution_approach')
            
            return alternatives[:5]  # Limit to top 5 alternatives
            
        except Exception as e:
            self.logger.error(f"Error generating alternative approaches: {e}")
            return ['fallback_manual_approach']

    async def _identify_optimization_opportunities(self, instruction: str,
                                                 planning_context: PlanningContext,
                                                 complexity_score: float) -> List[str]:
        """Identify opportunities for plan optimization"""
        try:
            opportunities = []
            
            # Complexity-based optimizations
            if complexity_score > 0.7:
                opportunities.append('task_decomposition_optimization')
                opportunities.append('parallel_execution_optimization')
            
            # Resource-based optimizations
            if len(planning_context.available_resources) > 0:
                opportunities.append('resource_utilization_optimization')
            
            # Experience-based optimizations
            if planning_context.previous_attempts:
                opportunities.append('learning_from_previous_attempts')
            
            # Domain-specific optimizations
            if planning_context.domain == 'mobile_app':
                opportunities.append('ui_element_caching_optimization')
            elif planning_context.domain == 'web':
                opportunities.append('page_load_optimization')
            
            # Quality-based optimizations
            if 'efficiency' in planning_context.quality_requirements:
                opportunities.append('execution_speed_optimization')
            if 'reliability' in planning_context.quality_requirements:
                opportunities.append('error_handling_optimization')
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying optimization opportunities: {e}")
            return []

    async def _create_comprehensive_plan(self, instruction: str, analysis: PlanAnalysis,
                                       attention_allocation: AttentionAllocation,
                                       context: Dict[str, Any] = None) -> QAPlan:
        """Create a comprehensive QA execution plan"""
        try:
            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            
            # Determine planning strategy
            strategy = await self._determine_planning_strategy(analysis)
            
            # Determine plan complexity
            complexity = self._determine_plan_complexity(analysis.complexity_score)
            
            # Generate execution steps
            execution_steps = await self._generate_execution_steps(
                instruction, analysis, strategy
            )
            
            # Create success criteria
            success_criteria = await self._create_success_criteria(instruction, analysis)
            
            # Create risk assessment
            risk_assessment = await self._create_risk_assessment(analysis)
            
            # Calculate total estimated time
            estimated_total_time = sum(step.estimated_duration for step in execution_steps)
            
            # Calculate confidence score
            confidence_score = await self._calculate_plan_confidence(analysis, execution_steps)
            
            # Create monitoring points
            monitoring_points = await self._create_monitoring_points(execution_steps)
            
            plan = QAPlan(
                plan_id=plan_id,
                goal=instruction,
                description=f"Comprehensive QA plan for: {instruction}",
                strategy=strategy,
                complexity=complexity,
                execution_steps=execution_steps,
                success_criteria=success_criteria,
                risk_assessment=risk_assessment,
                resource_requirements=analysis.resource_demands,
                estimated_total_time=estimated_total_time,
                confidence_score=confidence_score,
                created_timestamp=time.time(),
                created_by=self.agent_id,
                dependencies=[],
                alternatives=analysis.alternative_approaches,
                monitoring_points=monitoring_points
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive plan: {e}")
            raise

    async def _determine_planning_strategy(self, analysis: PlanAnalysis) -> PlanningStrategy:
        """Determine the optimal planning strategy"""
        try:
            complexity = analysis.complexity_score
            required_capabilities = analysis.required_capabilities
            
            # High complexity tasks benefit from hierarchical planning
            if complexity > 0.8:
                return PlanningStrategy.HIERARCHICAL
            
            # Tasks with many capabilities benefit from parallel execution
            elif len(required_capabilities) > 5:
                return PlanningStrategy.PARALLEL
            
            # Simple tasks can use sequential planning
            elif complexity < 0.4:
                return PlanningStrategy.SEQUENTIAL
            
            # Moderate complexity tasks benefit from adaptive planning
            elif len(analysis.risk_factors) > 2:
                return PlanningStrategy.ADAPTIVE
            
            # Default to incremental for balanced approach
            else:
                return PlanningStrategy.INCREMENTAL
                
        except Exception as e:
            self.logger.error(f"Error determining planning strategy: {e}")
            return PlanningStrategy.SEQUENTIAL

    def _determine_plan_complexity(self, complexity_score: float) -> PlanComplexity:
        """Determine plan complexity level"""
        if complexity_score > 0.8:
            return PlanComplexity.HIGHLY_COMPLEX
        elif complexity_score > 0.6:
            return PlanComplexity.COMPLEX
        elif complexity_score > 0.4:
            return PlanComplexity.MODERATE
        else:
            return PlanComplexity.SIMPLE

    async def _generate_execution_steps(self, instruction: str, analysis: PlanAnalysis,
                                       strategy: PlanningStrategy) -> List[QAPlanStep]:
        """Generate detailed execution steps"""
        try:
            steps = []
            
            # Use LLM to generate detailed steps
            llm_prompt = f"""
Create a detailed execution plan for the following task:
Task: {instruction}
Complexity: {analysis.complexity_score}
Required Capabilities: {', '.join(analysis.required_capabilities)}
Strategy: {strategy.value}

Generate 3-8 specific, actionable steps. For each step, provide:
1. Clear action description
2. Expected outcome
3. Prerequisites
4. Risk level (low/medium/high)
5. Estimated duration in seconds

Format as JSON list with these fields: action, description, prerequisites, expected_outcome, risk_level, estimated_duration
            """
            
            try:
                llm_response = await self.llm_interface.generate_response(
                    llm_prompt,
                    max_tokens=800,
                    temperature=0.3
                )
                
                # Parse LLM response
                steps_data = self._parse_llm_steps_response(llm_response)
                
            except Exception as e:
                self.logger.warning(f"LLM step generation failed: {e}, using fallback")
                steps_data = self._generate_fallback_steps(instruction, analysis)
            
            # Convert to QAPlanStep objects
            for i, step_data in enumerate(steps_data):
                step = QAPlanStep(
                    step_id=f"step_{i+1}_{uuid.uuid4().hex[:4]}",
                    step_number=i + 1,
                    action=step_data.get('action', f'Step {i+1}'),
                    description=step_data.get('description', 'Execute step'),
                    prerequisites=step_data.get('prerequisites', []),
                    expected_outcome=step_data.get('expected_outcome', 'Step completed'),
                    validation_criteria=[f"Verify {step_data.get('expected_outcome', 'completion')}"],
                    estimated_duration=step_data.get('estimated_duration', 30.0),
                    risk_level=step_data.get('risk_level', 'medium'),
                    fallback_options=self._generate_step_fallbacks(step_data),
                    attention_cost=self._calculate_step_attention_cost(step_data),
                    confidence=0.8
                )
                steps.append(step)
            
            return steps
            
        except Exception as e:
            self.logger.error(f"Error generating execution steps: {e}")
            return self._generate_fallback_steps(instruction, analysis)

    def _parse_llm_steps_response(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for execution steps"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                steps_data = json.loads(json_str)
                
                if isinstance(steps_data, list):
                    return steps_data
            
            # Fallback: parse text format
            return self._parse_text_steps_response(llm_response)
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM steps response: {e}")
            return []

    def _parse_text_steps_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse text format steps response"""
        steps = []
        lines = response.split('\n')
        
        current_step = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'action': line,
                    'description': line,
                    'prerequisites': [],
                    'expected_outcome': 'Step completed',
                    'risk_level': 'medium',
                    'estimated_duration': 30.0
                }
            elif current_step and line:
                # Add additional context to description
                current_step['description'] += f" {line}"
        
        if current_step:
            steps.append(current_step)
        
        return steps

    def _generate_fallback_steps(self, instruction: str, analysis: PlanAnalysis) -> List[QAPlanStep]:
        """Generate fallback steps when LLM fails"""
        steps = []
        
        # Basic steps based on task type
        basic_steps = [
            {
                'action': 'Initialize environment',
                'description': 'Set up the execution environment and verify prerequisites',
                'prerequisites': [],
                'expected_outcome': 'Environment ready for execution',
                'risk_level': 'low',
                'estimated_duration': 10.0
            },
            {
                'action': 'Execute main task',
                'description': f'Execute the main task: {instruction}',
                'prerequisites': ['Environment initialized'],
                'expected_outcome': 'Main task completed successfully',
                'risk_level': 'medium',
                'estimated_duration': 45.0
            },
            {
                'action': 'Verify results',
                'description': 'Verify that the task was completed successfully',
                'prerequisites': ['Main task executed'],
                'expected_outcome': 'Results verified and validated',
                'risk_level': 'low',
                'estimated_duration': 15.0
            }
        ]
        
        for i, step_data in enumerate(basic_steps):
            step = QAPlanStep(
                step_id=f"fallback_step_{i+1}",
                step_number=i + 1,
                action=step_data['action'],
                description=step_data['description'],
                prerequisites=step_data['prerequisites'],
                expected_outcome=step_data['expected_outcome'],
                validation_criteria=[f"Verify {step_data['expected_outcome']}"],
                estimated_duration=step_data['estimated_duration'],
                risk_level=step_data['risk_level'],
                fallback_options=[],
                attention_cost=2.0,
                confidence=0.7
            )
            steps.append(step)
        
        return steps

    def _generate_step_fallbacks(self, step_data: Dict[str, Any]) -> List[str]:
        """Generate fallback options for a step"""
        fallbacks = []
        
        risk_level = step_data.get('risk_level', 'medium')
        
        if risk_level == 'high':
            fallbacks.extend([
                'manual_verification_before_execution',
                'checkpoint_and_rollback_option',
                'alternative_execution_method'
            ])
        elif risk_level == 'medium':
            fallbacks.extend([
                'retry_with_different_parameters',
                'skip_and_continue_if_non_critical'
            ])
        else:
            fallbacks.append('standard_retry_mechanism')
        
        return fallbacks

    def _calculate_step_attention_cost(self, step_data: Dict[str, Any]) -> float:
        """Calculate attention cost for a step"""
        base_cost = 1.0
        
        # Adjust based on risk level
        risk_level = step_data.get('risk_level', 'medium')
        if risk_level == 'high':
            base_cost *= 1.5
        elif risk_level == 'low':
            base_cost *= 0.8
        
        # Adjust based on estimated duration
        duration = step_data.get('estimated_duration', 30.0)
        if duration > 60:
            base_cost *= 1.3
        elif duration < 15:
            base_cost *= 0.9
        
        return base_cost

    async def _create_success_criteria(self, instruction: str, analysis: PlanAnalysis) -> List[str]:
        """Create success criteria for the plan"""
        criteria = [
            'Task completed without errors',
            'All required capabilities successfully utilized',
            'Expected outcomes achieved within time constraints'
        ]
        
        # Add domain-specific criteria
        if 'ui_automation' in analysis.required_capabilities:
            criteria.append('UI interactions completed successfully')
        
        if 'data_processing' in analysis.required_capabilities:
            criteria.append('Data processed and validated correctly')
        
        if 'verification' in analysis.required_capabilities:
            criteria.append('Verification checks passed')
        
        # Add risk-specific criteria
        if 'destructive_operation_risk' in analysis.risk_factors:
            criteria.append('No unintended system modifications occurred')
        
        if 'network_dependency_risk' in analysis.risk_factors:
            criteria.append('Network operations completed successfully')
        
        return criteria

    async def _create_risk_assessment(self, analysis: PlanAnalysis) -> Dict[str, Any]:
        """Create comprehensive risk assessment"""
        risk_assessment = {
            'overall_risk_level': 'medium',
            'identified_risks': analysis.risk_factors,
            'risk_mitigation_strategies': [],
            'contingency_plans': [],
            'risk_probability': {},
            'risk_impact': {}
        }
        
        # Determine overall risk level
        if len(analysis.risk_factors) > 3 or any('high' in risk for risk in analysis.risk_factors):
            risk_assessment['overall_risk_level'] = 'high'
        elif len(analysis.risk_factors) < 2:
            risk_assessment['overall_risk_level'] = 'low'
        
        # Create mitigation strategies
        for risk in analysis.risk_factors:
            if 'complexity' in risk:
                risk_assessment['risk_mitigation_strategies'].append('Break down into smaller steps')
            elif 'time_pressure' in risk:
                risk_assessment['risk_mitigation_strategies'].append('Prioritize critical operations')
            elif 'resource' in risk:
                risk_assessment['risk_mitigation_strategies'].append('Optimize resource utilization')
        
        # Create contingency plans
        risk_assessment['contingency_plans'] = [
            'Fallback to manual execution if automation fails',
            'Checkpoint and rollback mechanism for critical steps',
            'Alternative approach selection based on intermediate results'
        ]
        
        return risk_assessment

    async def _calculate_plan_confidence(self, analysis: PlanAnalysis, 
                                       execution_steps: List[QAPlanStep]) -> float:
        """Calculate overall plan confidence score"""
        try:
            confidence_factors = []
            
            # Base confidence from success probability
            confidence_factors.append(analysis.success_probability)
            
            # Confidence from step reliability
            step_confidences = [step.confidence for step in execution_steps]
            if step_confidences:
                confidence_factors.append(sum(step_confidences) / len(step_confidences))
            
            # Confidence adjustment based on risk factors
            risk_penalty = len(analysis.risk_factors) * 0.05
            confidence_factors.append(max(0.1, 1.0 - risk_penalty))
            
            # Confidence from resource availability
            if analysis.resource_demands:
                resource_confidence = min(1.0, len(analysis.resource_demands) * 0.1)
                confidence_factors.append(resource_confidence)
            
            # Calculate weighted average
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating plan confidence: {e}")
            return 0.7

    async def _create_monitoring_points(self, execution_steps: List[QAPlanStep]) -> List[str]:
        """Create monitoring points for plan execution"""
        monitoring_points = []
        
        # Add monitoring after critical steps
        for step in execution_steps:
            if step.risk_level == 'high':
                monitoring_points.append(f"Monitor after {step.action}")
            
            if step.estimated_duration > 60:
                monitoring_points.append(f"Progress check during {step.action}")
        
        # Add standard monitoring points
        monitoring_points.extend([
            'Verify environment setup',
            'Check intermediate results',
            'Validate final outcomes'
        ])
        
        return monitoring_points

    async def _optimize_plan(self, plan: QAPlan, analysis: PlanAnalysis) -> QAPlan:
        """Optimize the plan based on analysis"""
        try:
            optimized_plan = plan
            
            # Apply optimization opportunities
            for opportunity in analysis.optimization_opportunities:
                if opportunity == 'task_decomposition_optimization':
                    optimized_plan = await self._apply_task_decomposition_optimization(optimized_plan)
                elif opportunity == 'parallel_execution_optimization':
                    optimized_plan = await self._apply_parallel_execution_optimization(optimized_plan)
                elif opportunity == 'resource_utilization_optimization':
                    optimized_plan = await self._apply_resource_optimization(optimized_plan)
            
            # Update confidence score after optimization
            optimized_plan.confidence_score = min(1.0, optimized_plan.confidence_score + 0.05)
            
            return optimized_plan
            
        except Exception as e:
            self.logger.error(f"Error optimizing plan: {e}")
            return plan

    async def _apply_task_decomposition_optimization(self, plan: QAPlan) -> QAPlan:
        """Apply task decomposition optimization"""
        # For complex steps, break them down further
        optimized_steps = []
        
        for step in plan.execution_steps:
            if step.estimated_duration > 90 and step.risk_level == 'high':
                # Break down into sub-steps
                sub_steps = self._decompose_step(step)
                optimized_steps.extend(sub_steps)
            else:
                optimized_steps.append(step)
        
        plan.execution_steps = optimized_steps
        plan.estimated_total_time = sum(step.estimated_duration for step in optimized_steps)
        
        return plan

    def _decompose_step(self, step: QAPlanStep) -> List[QAPlanStep]:
        """Decompose a complex step into sub-steps"""
        sub_steps = []
        
        # Create preparation step
        prep_step = QAPlanStep(
            step_id=f"{step.step_id}_prep",
            step_number=step.step_number,
            action=f"Prepare for {step.action}",
            description=f"Preparation phase for {step.description}",
            prerequisites=step.prerequisites,
            expected_outcome=f"Ready to execute {step.action}",
            validation_criteria=[f"Prerequisites verified for {step.action}"],
            estimated_duration=step.estimated_duration * 0.2,
            risk_level='low',
            fallback_options=[],
            attention_cost=step.attention_cost * 0.3,
            confidence=0.9
        )
        sub_steps.append(prep_step)
        
        # Create main execution step
        main_step = QAPlanStep(
            step_id=f"{step.step_id}_main",
            step_number=step.step_number,
            action=step.action,
            description=step.description,
            prerequisites=[prep_step.expected_outcome],
            expected_outcome=step.expected_outcome,
            validation_criteria=step.validation_criteria,
            estimated_duration=step.estimated_duration * 0.6,
            risk_level=step.risk_level,
            fallback_options=step.fallback_options,
            attention_cost=step.attention_cost * 0.6,
            confidence=step.confidence
        )
        sub_steps.append(main_step)
        
        # Create verification step
        verify_step = QAPlanStep(
            step_id=f"{step.step_id}_verify",
            step_number=step.step_number,
            action=f"Verify {step.action}",
            description=f"Verification phase for {step.description}",
            prerequisites=[main_step.expected_outcome],
            expected_outcome=f"Verified {step.expected_outcome}",
            validation_criteria=[f"Confirmed {step.expected_outcome}"],
            estimated_duration=step.estimated_duration * 0.2,
            risk_level='low',
            fallback_options=[],
            attention_cost=step.attention_cost * 0.1,
            confidence=0.95
        )
        sub_steps.append(verify_step)
        
        return sub_steps

    async def _apply_parallel_execution_optimization(self, plan: QAPlan) -> QAPlan:
        """Apply parallel execution optimization"""
        # Identify steps that can be executed in parallel
        # For now, this is a simplified implementation
        # In a full implementation, this would analyze dependencies
        
        plan.strategy = PlanningStrategy.PARALLEL
        
        return plan

    async def _apply_resource_optimization(self, plan: QAPlan) -> QAPlan:
        """Apply resource utilization optimization"""
        # Optimize resource allocation across steps
        total_attention = sum(step.attention_cost for step in plan.execution_steps)
        
        if total_attention > 10.0:  # High attention requirement
            # Reduce attention cost by optimizing step order
            for step in plan.execution_steps:
                if step.risk_level == 'low':
                    step.attention_cost *= 0.9
        
        # Update resource requirements
        plan.resource_requirements['attention'] = total_attention * 0.95
        
        return plan

    async def _create_planning_actions(self, plan: QAPlan) -> List[str]:
        """Create planning actions from the plan"""
        actions = []
        
        try:
            # Main planning action
            actions.append(f"# Plan created: {plan.plan_id}")
            actions.append(f"# Strategy: {plan.strategy.value}")
            actions.append(f"# Complexity: {plan.complexity.value}")
            actions.append(f"# Steps: {len(plan.execution_steps)}")
            actions.append(f"# Estimated time: {plan.estimated_total_time:.1f}s")
            actions.append(f"# Confidence: {plan.confidence_score:.2f}")
            
            # Step summaries
            for i, step in enumerate(plan.execution_steps[:5]):  # Show first 5 steps
                actions.append(f"# Step {i+1}: {step.action}")
            
            if len(plan.execution_steps) > 5:
                actions.append(f"# ... and {len(plan.execution_steps) - 5} more steps")
            
            # Risk assessment summary
            risk_level = plan.risk_assessment.get('overall_risk_level', 'medium')
            actions.append(f"# Risk level: {risk_level}")
            
            # Success criteria summary
            actions.append(f"# Success criteria: {len(plan.success_criteria)} defined")
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error creating planning actions: {e}")
            return [f"# Plan created with errors: {str(e)}"]

    async def _update_planning_metrics(self, plan: QAPlan, attention_allocation: AttentionAllocation):
        """Update planning performance metrics"""
        try:
            self.planning_metrics['plans_created'] += 1
            
            # Update complexity handled
            complexity_score = 0.5
            if plan.complexity == PlanComplexity.SIMPLE:
                complexity_score = 0.3
            elif plan.complexity == PlanComplexity.MODERATE:
                complexity_score = 0.5
            elif plan.complexity == PlanComplexity.COMPLEX:
                complexity_score = 0.7
            elif plan.complexity == PlanComplexity.HIGHLY_COMPLEX:
                complexity_score = 0.9
            
            current_complexity = self.planning_metrics['complexity_handled']
            plans_count = self.planning_metrics['plans_created']
            
            self.planning_metrics['complexity_handled'] = (
                (current_complexity * (plans_count - 1) + complexity_score) / plans_count
            )
            
            # Update planning time (simulated)
            planning_time = attention_allocation.total_attention * 2.0  # Rough estimate
            current_avg_time = self.planning_metrics['average_planning_time']
            
            self.planning_metrics['average_planning_time'] = (
                (current_avg_time * (plans_count - 1) + planning_time) / plans_count
            )
            
            # Assume plan will be successful (updated later during execution)
            self.planning_metrics['successful_plans'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating planning metrics: {e}")

    async def _load_planning_templates(self):
        """Load planning templates"""
        try:
            # In a full implementation, this would load from storage
            self.plan_templates = {
                'ui_automation': {
                    'steps': ['setup', 'navigate', 'interact', 'verify'],
                    'risks': ['element_not_found', 'timing_issues'],
                    'resources': {'attention': 3.0, 'time': 60.0}
                },
                'data_processing': {
                    'steps': ['extract', 'transform', 'validate', 'store'],
                    'risks': ['data_corruption', 'format_mismatch'],
                    'resources': {'attention': 4.0, 'time': 90.0}
                }
            }
            
        except Exception as e:
            self.logger.debug(f"No existing planning templates to load: {e}")

    async def _load_domain_knowledge(self):
        """Load domain-specific knowledge"""
        try:
            self.domain_knowledge = {
                'mobile_app': [
                    'Touch gestures are primary interaction method',
                    'UI elements may vary between devices',
                    'App state can change between interactions'
                ],
                'web': [
                    'Page load times can vary',
                    'Elements may be dynamically loaded',
                    'Cross-browser compatibility considerations'
                ],
                'system': [
                    'Permissions may be required',
                    'System state affects operations',
                    'Settings changes may require restart'
                ]
            }
            
        except Exception as e:
            self.logger.debug(f"Error loading domain knowledge: {e}")

    async def _load_strategy_patterns(self):
        """Load strategy patterns"""
        try:
            self.strategy_patterns = {
                'sequential': [
                    {'pattern': 'linear_execution', 'success_rate': 0.8},
                    {'pattern': 'checkpoint_validation', 'success_rate': 0.85}
                ],
                'parallel': [
                    {'pattern': 'independent_tasks', 'success_rate': 0.75},
                    {'pattern': 'synchronized_completion', 'success_rate': 0.7}
                ]
            }
            
        except Exception as e:
            self.logger.debug(f"Error loading strategy patterns: {e}")

    async def _save_planning_knowledge(self):
        """Save planning knowledge"""
        try:
            # In a full implementation, this would save to persistent storage
            pass
            
        except Exception as e:
            self.logger.error(f"Error saving planning knowledge: {e}")

    def get_planning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive planning analytics"""
        try:
            base_analytics = super().get_execution_analytics()
            
            # Add planner-specific analytics
            planner_analytics = {
                'planning_metrics': self.planning_metrics.copy(),
                'active_plans_count': len(self.active_plans),
                'plan_templates_count': len(self.plan_templates),
                'domain_knowledge_areas': len(self.domain_knowledge),
                'strategy_patterns_count': sum(len(patterns) for patterns in self.strategy_patterns.values()),
                'planning_history_size': len(self.planning_history),
                
                # Planning performance
                'success_rate': (
                    self.planning_metrics['successful_plans'] / 
                    max(1, self.planning_metrics['plans_created'])
                ),
                'average_plan_complexity': self.planning_metrics['complexity_handled'],
                'planning_efficiency': (
                    60.0 / max(1.0, self.planning_metrics['average_planning_time'])  # Plans per minute
                )
            }
            
            # Merge with base analytics
            base_analytics.update(planner_analytics)
            
            return base_analytics
            
        except Exception as e:
            self.logger.error(f"Error generating planning analytics: {e}")
            return {'error': str(e)}

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get status of specific plan"""
        try:
            if plan_id in self.active_plans:
                plan = self.active_plans[plan_id]
                return {
                    'plan_id': plan_id,
                    'status': 'active',
                    'goal': plan.goal,
                    'strategy': plan.strategy.value,
                    'complexity': plan.complexity.value,
                    'steps_count': len(plan.execution_steps),
                    'confidence_score': plan.confidence_score,
                    'estimated_time': plan.estimated_total_time,
                    'created_timestamp': plan.created_timestamp
                }
            
            # Check planning history
            for plan in self.planning_history:
                if plan.plan_id == plan_id:
                    return {
                        'plan_id': plan_id,
                        'status': 'archived',
                        'goal': plan.goal,
                        'strategy': plan.strategy.value,
                        'complexity': plan.complexity.value,
                        'confidence_score': plan.confidence_score
                    }
            
            return {'plan_id': plan_id, 'status': 'not_found'}
            
        except Exception as e:
            self.logger.error(f"Error getting plan status: {e}")
            return {'plan_id': plan_id, 'status': 'error', 'error': str(e)}


__all__ = [
    "PlannerAgent",
    "QAPlan",
    "QAPlanStep",
    "PlanningStrategy",
    "PlanComplexity",
    "PlanningContext",
    "PlanAnalysis"
]