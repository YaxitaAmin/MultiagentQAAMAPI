"""
Supervisor Agent for AMAPI System
Orchestrates multiple agents and manages overall system coordination
"""

import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from agents.base_agent import BaseQAAgent, AgentType, AgentAction, AgentObservation, AgentMessage, MessageType
from agents.planner_agent import PlannerAgent, QAPlan
from agents.executor_agent import ExecutorAgent, ExecutionResult
from core.attention_economics import AttentionEconomicsEngine
from core.behavioral_learning import BehavioralPatternEngine, LearningType
from core.llm_interface import LLMInterface
from utils.attention_helpers import calculate_supervision_attention_cost


class SupervisionMode(Enum):
    """Supervision modes for different scenarios"""
    AUTONOMOUS = "autonomous"
    GUIDED = "guided"
    COLLABORATIVE = "collaborative"
    EMERGENCY = "emergency"


class SystemState(Enum):
    """Overall system states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    COORDINATING = "coordinating"
    LEARNING = "learning"
    ERROR_RECOVERY = "error_recovery"
    SHUTDOWN = "shutdown"


@dataclass
class SupervisionTask:
    """High-level supervision task"""
    task_id: str
    description: str
    agents_involved: List[str]
    priority: int
    status: str
    created_timestamp: float
    deadline: Optional[float] = None
    dependencies: List[str] = None
    success_criteria: List[str] = None
    current_plan: Optional[str] = None
    execution_results: List[str] = None


@dataclass
class AgentCoordination:
    """Agent coordination information"""
    coordination_id: str
    primary_agent: str
    supporting_agents: List[str]
    coordination_type: str
    task_distribution: Dict[str, List[str]]
    communication_protocol: str
    success_metrics: Dict[str, float]
    timestamp: float


@dataclass
class SystemHealthStatus:
    """System health status"""
    overall_health: str
    agent_statuses: Dict[str, str]
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, float]
    active_issues: List[str]
    recommendations: List[str]
    timestamp: float


class SupervisorAgent(BaseQAAgent):
    """
    Advanced Supervisor Agent with AMAPI Integration
    Orchestrates multi-agent collaboration and manages system-wide coordination
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            agent_type=AgentType.SUPERVISOR,
            config=config
        )
        
        # Managed agents
        self.managed_agents: Dict[str, BaseQAAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_workloads: Dict[str, float] = {}
        
        # Supervision components
        self.llm_interface = LLMInterface(config.get('llm', {}))
        self.behavioral_engine = BehavioralPatternEngine(config.get('behavioral', {}))
        
        # System state management
        self.system_state = SystemState.INITIALIZING
        self.supervision_mode = SupervisionMode.AUTONOMOUS
        self.active_tasks: Dict[str, SupervisionTask] = {}
        self.task_history: List[SupervisionTask] = []
        
        # Coordination management
        self.active_coordinations: Dict[str, AgentCoordination] = {}
        self.coordination_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # System monitoring
        self.system_health: SystemHealthStatus = None
        self.performance_trends: List[Dict[str, Any]] = []
        self.alert_thresholds: Dict[str, float] = {
            'agent_response_time': 10.0,
            'success_rate': 0.7,
            'resource_utilization': 0.9,
            'error_rate': 0.3
        }
        
        # Supervisor specializations
        self.specializations = [
            'multi_agent_coordination',
            'task_orchestration',
            'system_monitoring',
            'resource_optimization',
            'conflict_resolution',
            'performance_analysis'
        ]
        
        # Supervision metrics
        self.supervision_metrics = {
            'tasks_supervised': 0,
            'successful_coordinations': 0,
            'agents_managed': 0,
            'system_uptime': 0.0,
            'average_task_completion_time': 0.0,
            'coordination_efficiency': 0.0,
            'conflict_resolutions': 0,
            'performance_optimizations': 0
        }
        
        self.logger.info("Supervisor Agent initialized")

    async def _initialize_agent_systems(self) -> None:
        """Initialize supervisor-specific systems"""
        try:
            # Initialize LLM interface
            await self.llm_interface.initialize()
            
            # Start behavioral learning engine
            await self.behavioral_engine.start_learning_engine()
            
            # Initialize system monitoring
            await self._initialize_system_monitoring()
            
            # Load coordination patterns
            await self._load_coordination_patterns()
            
            # Set system state to idle
            self.system_state = SystemState.IDLE
            
            self.logger.info("Supervisor agent systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize supervisor systems: {e}")
            raise

    async def _cleanup_agent_systems(self) -> None:
        """Cleanup supervisor-specific systems"""
        try:
            await self.llm_interface.cleanup()
            await self.behavioral_engine.stop_learning_engine()
            await self._save_coordination_patterns()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up supervisor systems: {e}")

    async def predict(self, instruction: str, observation: Dict[str, Any], 
                     context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Supervise and coordinate agent activities"""
        try:
            self.system_state = SystemState.COORDINATING
            
            # Allocate attention for supervision
            attention_allocation = await self.allocate_attention(
                f"Supervise: {instruction}",
                context.get('task_complexity', 0.6) if context else 0.6
            )
            
            # Analyze supervision requirements
            supervision_analysis = await self._analyze_supervision_requirements(
                instruction, observation, context
            )
            
            # Determine optimal coordination strategy
            coordination_strategy = await self._determine_coordination_strategy(
                supervision_analysis, attention_allocation
            )
            
            # Execute supervision
            supervision_result = await self._execute_supervision(
                coordination_strategy, instruction, observation, context
            )
            
            # Create supervision actions
            actions = await self._create_supervision_actions(supervision_result)
            
            # Update supervision metrics
            await self._update_supervision_metrics(supervision_result)
            
            # Create reasoning info
            reasoning_info = {
                'supervision_analysis': supervision_analysis,
                'coordination_strategy': coordination_strategy,
                'supervision_result': supervision_result,
                'agents_coordinated': len(supervision_result.get('coordinated_agents', [])),
                'tasks_orchestrated': len(supervision_result.get('orchestrated_tasks', [])),
                'system_health_score': supervision_result.get('system_health_score', 0.8),
                'coordination_efficiency': supervision_result.get('coordination_efficiency', 0.8),
                'attention_utilization': attention_allocation.efficiency_score,
                'supervision_mode': self.supervision_mode.value,
                'system_state': self.system_state.value
            }
            
            self.system_state = SystemState.IDLE
            self.logger.info(f"Supervision completed for: {instruction}")
            
            return reasoning_info, actions
            
        except Exception as e:
            self.system_state = SystemState.ERROR_RECOVERY
            self.logger.error(f"Error in supervisor prediction: {e}")
            return {'error': str(e), 'confidence': 0.0}, []

    async def _analyze_supervision_requirements(self, instruction: str, observation: Dict[str, Any],
                                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze what kind of supervision is needed"""
        try:
            analysis = {
                'task_complexity': 'medium',
                'agents_required': [],
                'coordination_type': 'sequential',
                'resource_requirements': {},
                'risk_factors': [],
                'success_criteria': [],
                'estimated_duration': 30.0
            }
            
            # Classify task type
            task_type = self._classify_supervision_task(instruction)
            analysis['task_type'] = task_type
            
            # Determine required agents based on task
            if task_type == 'planning_supervision':
                analysis['agents_required'] = ['planner']
                analysis['coordination_type'] = 'sequential'
            elif task_type == 'execution_supervision':
                analysis['agents_required'] = ['executor']
                analysis['coordination_type'] = 'monitored'
            elif task_type == 'complex_task_supervision':
                analysis['agents_required'] = ['planner', 'executor', 'verifier']
                analysis['coordination_type'] = 'pipeline'
            elif task_type == 'collaborative_supervision':
                analysis['agents_required'] = ['planner', 'executor']
                analysis['coordination_type'] = 'collaborative'
            
            # Assess complexity
            complexity_indicators = [
                len(analysis['agents_required']),
                len(instruction.split()) / 10,
                context.get('task_complexity', 0.5) if context else 0.5
            ]
            complexity_score = min(1.0, sum(complexity_indicators) / len(complexity_indicators))
            
            if complexity_score > 0.8:
                analysis['task_complexity'] = 'high'
            elif complexity_score > 0.5:
                analysis['task_complexity'] = 'medium'
            else:
                analysis['task_complexity'] = 'low'
            
            # Identify resource requirements
            analysis['resource_requirements'] = {
                'attention_budget': complexity_score * 5.0,
                'time_budget': complexity_score * 60.0,  # seconds
                'coordination_overhead': len(analysis['agents_required']) * 0.2
            }
            
            # Identify potential risks
            analysis['risk_factors'] = self._identify_supervision_risks(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing supervision requirements: {e}")
            return {'task_type': 'general_supervision', 'agents_required': [], 'error': str(e)}

    def _classify_supervision_task(self, instruction: str) -> str:
        """Classify the type of supervision task"""
        instruction_lower = instruction.lower()
        
        if any(word in instruction_lower for word in ['plan', 'strategy', 'design', 'analyze']):
            return 'planning_supervision'
        elif any(word in instruction_lower for word in ['execute', 'perform', 'run', 'action']):
            return 'execution_supervision'
        elif any(word in instruction_lower for word in ['coordinate', 'collaborate', 'team', 'multiple']):
            return 'collaborative_supervision'
        elif any(word in instruction_lower for word in ['complex', 'multi-step', 'pipeline', 'workflow']):
            return 'complex_task_supervision'
        elif any(word in instruction_lower for word in ['monitor', 'watch', 'track', 'observe']):
            return 'monitoring_supervision'
        else:
            return 'general_supervision'

    def _identify_supervision_risks(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify potential risks in supervision"""
        risks = []
        
        # Agent availability risks
        if len(analysis['agents_required']) > len(self.managed_agents):
            risks.append('insufficient_agents')
        
        # Resource constraint risks
        if analysis['resource_requirements']['attention_budget'] > 10.0:
            risks.append('high_attention_requirement')
        
        # Coordination complexity risks
        if analysis['coordination_type'] in ['pipeline', 'collaborative']:
            risks.append('complex_coordination')
        
        # Time constraint risks
        if analysis['resource_requirements']['time_budget'] > 300:  # 5 minutes
            risks.append('long_execution_time')
        
        return risks

    async def _determine_coordination_strategy(self, analysis: Dict[str, Any],
                                             attention_allocation) -> Dict[str, Any]:
        """Determine optimal coordination strategy"""
        try:
            strategy = {
                'coordination_pattern': 'sequential',
                'agent_assignments': {},
                'communication_protocol': 'message_passing',
                'monitoring_level': 'standard',
                'fallback_plans': [],
                'optimization_targets': ['efficiency', 'success_rate']
            }
            
            # Select coordination pattern based on analysis
            coordination_type = analysis.get('coordination_type', 'sequential')
            
            if coordination_type == 'sequential':
                strategy['coordination_pattern'] = 'sequential'
                strategy['communication_protocol'] = 'handoff'
            elif coordination_type == 'pipeline':
                strategy['coordination_pattern'] = 'pipeline'
                strategy['communication_protocol'] = 'streaming'
            elif coordination_type == 'collaborative':
                strategy['coordination_pattern'] = 'collaborative'
                strategy['communication_protocol'] = 'broadcast'
            elif coordination_type == 'monitored':
                strategy['coordination_pattern'] = 'monitored'
                strategy['monitoring_level'] = 'high'
            
            # Assign agents to tasks
            required_agents = analysis.get('agents_required', [])
            available_agents = self._get_available_agents()
            
            for agent_type in required_agents:
                suitable_agents = [
                    agent_id for agent_id, agent in available_agents.items()
                    if agent_type in agent.specializations or agent.agent_type.value == agent_type
                ]
                
                if suitable_agents:
                    # Select best agent based on workload and capabilities
                    best_agent = min(suitable_agents, key=lambda x: self.agent_workloads.get(x, 0.0))
                    strategy['agent_assignments'][agent_type] = best_agent
                else:
                    self.logger.warning(f"No suitable agent found for type: {agent_type}")
            
            # Set monitoring level based on risk
            risk_factors = analysis.get('risk_factors', [])
            if any(risk in ['complex_coordination', 'high_attention_requirement'] for risk in risk_factors):
                strategy['monitoring_level'] = 'high'
            
            # Create fallback plans
            strategy['fallback_plans'] = await self._create_fallback_plans(analysis, strategy)
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error determining coordination strategy: {e}")
            return {'coordination_pattern': 'sequential', 'error': str(e)}

    def _get_available_agents(self) -> Dict[str, BaseQAAgent]:
        """Get available agents for task assignment"""
        available = {}
        
        for agent_id, agent in self.managed_agents.items():
            if agent.is_running and agent.state.value not in ['error', 'shutdown']:
                # Check workload
                current_workload = self.agent_workloads.get(agent_id, 0.0)
                if current_workload < 0.8:  # Not overloaded
                    available[agent_id] = agent
        
        return available

    async def _create_fallback_plans(self, analysis: Dict[str, Any], 
                                   strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback plans for supervision"""
        fallback_plans = []
        
        try:
            # Fallback for agent unavailability
            if 'insufficient_agents' in analysis.get('risk_factors', []):
                fallback_plans.append({
                    'trigger': 'agent_unavailable',
                    'action': 'sequential_execution',
                    'description': 'Execute tasks sequentially with available agents'
                })
            
            # Fallback for high complexity
            if analysis.get('task_complexity') == 'high':
                fallback_plans.append({
                    'trigger': 'complexity_exceeded',
                    'action': 'task_decomposition',
                    'description': 'Break down complex task into simpler subtasks'
                })
            
            # Fallback for coordination failure
            if strategy.get('coordination_pattern') in ['pipeline', 'collaborative']:
                fallback_plans.append({
                    'trigger': 'coordination_failure',
                    'action': 'switch_to_sequential',
                    'description': 'Switch to sequential coordination pattern'
                })
            
            return fallback_plans
            
        except Exception as e:
            self.logger.error(f"Error creating fallback plans: {e}")
            return []

    async def _execute_supervision(self, strategy: Dict[str, Any], instruction: str,
                                 observation: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the supervision strategy"""
        try:
            supervision_result = {
                'success': False,
                'coordinated_agents': [],
                'orchestrated_tasks': [],
                'coordination_efficiency': 0.0,
                'system_health_score': 0.8,
                'execution_timeline': [],
                'performance_metrics': {},
                'issues_encountered': [],
                'lessons_learned': []
            }
            
            coordination_pattern = strategy.get('coordination_pattern', 'sequential')
            agent_assignments = strategy.get('agent_assignments', {})
            
            if coordination_pattern == 'sequential':
                result = await self._execute_sequential_coordination(
                    agent_assignments, instruction, observation, context
                )
            elif coordination_pattern == 'pipeline':
                result = await self._execute_pipeline_coordination(
                    agent_assignments, instruction, observation, context
                )
            elif coordination_pattern == 'collaborative':
                result = await self._execute_collaborative_coordination(
                    agent_assignments, instruction, observation, context
                )
            elif coordination_pattern == 'monitored':
                result = await self._execute_monitored_coordination(
                    agent_assignments, instruction, observation, context
                )
            else:
                result = await self._execute_default_coordination(
                    agent_assignments, instruction, observation, context
                )
            
            supervision_result.update(result)
            
            # Calculate coordination efficiency
            supervision_result['coordination_efficiency'] = self._calculate_coordination_efficiency(
                supervision_result
            )
            
            # Update system health
            supervision_result['system_health_score'] = await self._assess_system_health()
            
            # Record learning event
            await self.behavioral_engine.record_learning_event(
                self.agent_id,
                LearningType.COLLABORATION,
                {
                    'coordination_pattern': coordination_pattern,
                    'agents_involved': list(agent_assignments.values()),
                    'success': supervision_result['success'],
                    'efficiency': supervision_result['coordination_efficiency']
                },
                'success' if supervision_result['success'] else 'failure',
                {'efficiency': supervision_result['coordination_efficiency']}
            )
            
            return supervision_result
            
        except Exception as e:
            self.logger.error(f"Error executing supervision: {e}")
            return {
                'success': False,
                'error': str(e),
                'coordination_efficiency': 0.0,
                'system_health_score': 0.5
            }

    async def _execute_sequential_coordination(self, agent_assignments: Dict[str, str],
                                             instruction: str, observation: Dict[str, Any],
                                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute sequential coordination pattern"""
        try:
            result = {
                'success': True,
                'coordinated_agents': [],
                'orchestrated_tasks': [],
                'execution_timeline': []
            }
            
            # Execute agents in sequence
            current_observation = observation
            
            for agent_type, agent_id in agent_assignments.items():
                if agent_id in self.managed_agents:
                    agent = self.managed_agents[agent_id]
                    
                    start_time = time.time()
                    
                    # Execute agent task
                    agent_result, actions = await agent.predict(
                        instruction, current_observation, context
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Record execution
                    result['coordinated_agents'].append(agent_id)
                    result['orchestrated_tasks'].append({
                        'agent_id': agent_id,
                        'agent_type': agent_type,
                        'success': agent_result.get('success', True),
                        'execution_time': execution_time,
                        'actions_count': len(actions)
                    })
                    
                    result['execution_timeline'].append({
                        'agent_id': agent_id,
                        'start_time': start_time,
                        'end_time': time.time(),
                        'duration': execution_time
                    })
                    
                    # Update observation for next agent
                    if 'observation' in agent_result:
                        current_observation = agent_result['observation']
                    
                    # Check for failure
                    if not agent_result.get('success', True):
                        result['success'] = False
                        break
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in sequential coordination: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_collaborative_coordination(self, agent_assignments: Dict[str, str],
                                                instruction: str, observation: Dict[str, Any],
                                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute collaborative coordination pattern"""
        try:
            result = {
                'success': True,
                'coordinated_agents': [],
                'orchestrated_tasks': [],
                'execution_timeline': []
            }
            
            # Execute agents in parallel with collaboration
            tasks = []
            
            for agent_type, agent_id in agent_assignments.items():
                if agent_id in self.managed_agents:
                    agent = self.managed_agents[agent_id]
                    
                    # Create collaborative context
                    collaborative_context = {
                        **(context or {}),
                        'collaboration_mode': True,
                        'partner_agents': [aid for aid in agent_assignments.values() if aid != agent_id]
                    }
                    
                    # Create task for parallel execution
                    task = asyncio.create_task(
                        agent.predict(instruction, observation, collaborative_context)
                    )
                    tasks.append((agent_id, agent_type, task))
            
            # Wait for all agents to complete
            start_time = time.time()
            
            for agent_id, agent_type, task in tasks:
                try:
                    agent_result, actions = await task
                    
                    result['coordinated_agents'].append(agent_id)
                    result['orchestrated_tasks'].append({
                        'agent_id': agent_id,
                        'agent_type': agent_type,
                        'success': agent_result.get('success', True),
                        'actions_count': len(actions)
                    })
                    
                    if not agent_result.get('success', True):
                        result['success'] = False
                
                except Exception as e:
                    self.logger.error(f"Error in collaborative agent {agent_id}: {e}")
                    result['success'] = False
            
            execution_time = time.time() - start_time
            result['execution_timeline'].append({
                'coordination_type': 'parallel',
                'total_duration': execution_time,
                'agents_count': len(tasks)
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in collaborative coordination: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_pipeline_coordination(self, agent_assignments: Dict[str, str],
                                           instruction: str, observation: Dict[str, Any],
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute pipeline coordination pattern"""
        try:
            # For now, pipeline coordination is similar to sequential
            # but with streaming data between agents
            return await self._execute_sequential_coordination(
                agent_assignments, instruction, observation, context
            )
        except Exception as e:
            self.logger.error(f"Error in pipeline coordination: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_monitored_coordination(self, agent_assignments: Dict[str, str],
                                            instruction: str, observation: Dict[str, Any],
                                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute monitored coordination pattern"""
        try:
            result = {
                'success': True,
                'coordinated_agents': [],
                'orchestrated_tasks': [],
                'monitoring_events': []
            }
            
            # Execute with enhanced monitoring
            for agent_type, agent_id in agent_assignments.items():
                if agent_id in self.managed_agents:
                    agent = self.managed_agents[agent_id]
                    
                    # Monitor agent execution
                    monitoring_context = {
                        **(context or {}),
                        'monitoring_enabled': True,
                        'supervisor_agent_id': self.agent_id
                    }
                    
                    start_time = time.time()
                    
                    # Execute with monitoring
                    agent_result, actions = await agent.predict(
                        instruction, observation, monitoring_context
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Record monitoring data
                    result['coordinated_agents'].append(agent_id)
                    result['orchestrated_tasks'].append({
                        'agent_id': agent_id,
                        'agent_type': agent_type,
                        'success': agent_result.get('success', True),
                        'execution_time': execution_time,
                        'performance_score': agent_result.get('performance_score', 0.8)
                    })
                    
                    result['monitoring_events'].append({
                        'agent_id': agent_id,
                        'timestamp': time.time(),
                        'status': 'completed',
                        'metrics': {
                            'execution_time': execution_time,
                            'success': agent_result.get('success', True)
                        }
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in monitored coordination: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_default_coordination(self, agent_assignments: Dict[str, str],
                                          instruction: str, observation: Dict[str, Any],
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute default coordination pattern"""
        return await self._execute_sequential_coordination(
            agent_assignments, instruction, observation, context
        )

    def _calculate_coordination_efficiency(self, supervision_result: Dict[str, Any]) -> float:
        """Calculate coordination efficiency score"""
        try:
            # Base efficiency on success rate
            successful_tasks = sum(
                1 for task in supervision_result.get('orchestrated_tasks', [])
                if task.get('success', False)
            )
            total_tasks = len(supervision_result.get('orchestrated_tasks', []))
            
            if total_tasks == 0:
                return 0.5
            
            success_rate = successful_tasks / total_tasks
            
            # Adjust for coordination overhead
            agents_count = len(supervision_result.get('coordinated_agents', []))
            coordination_overhead = min(0.3, agents_count * 0.05)
            
            efficiency = success_rate - coordination_overhead
            
            return max(0.0, min(1.0, efficiency))
            
        except Exception as e:
            self.logger.error(f"Error calculating coordination efficiency: {e}")
            return 0.5

    async def _assess_system_health(self) -> float:
        """Assess overall system health"""
        try:
            health_factors = []
            
            # Agent health
            healthy_agents = sum(
                1 for agent in self.managed_agents.values()
                if agent.is_running and agent.state.value not in ['error', 'shutdown']
            )
            total_agents = len(self.managed_agents)
            
            if total_agents > 0:
                agent_health = healthy_agents / total_agents
                health_factors.append(agent_health * 0.4)
            
            # Resource utilization
            avg_workload = sum(self.agent_workloads.values()) / max(1, len(self.agent_workloads))
            resource_health = 1.0 - min(1.0, avg_workload)
            health_factors.append(resource_health * 0.3)
            
            # Performance metrics
            success_rate = (
                self.supervision_metrics['successful_coordinations'] /
                max(1, self.supervision_metrics['tasks_supervised'])
            )
            health_factors.append(success_rate * 0.3)
            
            return sum(health_factors) if health_factors else 0.8
            
        except Exception as e:
            self.logger.error(f"Error assessing system health: {e}")
            return 0.5

    async def _create_supervision_actions(self, supervision_result: Dict[str, Any]) -> List[str]:
        """Create supervision actions from result"""
        actions = []
        
        try:
            # Main supervision action
            if supervision_result.get('success', False):
                actions.append(f"# Supervision completed successfully")
                actions.append(f"# Coordinated {len(supervision_result.get('coordinated_agents', []))} agents")
            else:
                actions.append(f"# Supervision completed with issues")
                actions.append(f"# Review and retry coordination")
            
            # Agent coordination actions
            for task in supervision_result.get('orchestrated_tasks', []):
                agent_id = task.get('agent_id', 'unknown')
                success = task.get('success', False)
                actions.append(f"# Agent {agent_id}: {'SUCCESS' if success else 'FAILED'}")
            
            # System health actions
            health_score = supervision_result.get('system_health_score', 0.8)
            if health_score < 0.7:
                actions.append(f"# ALERT: System health low ({health_score:.2f})")
                actions.append(f"# Recommend system optimization")
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error creating supervision actions: {e}")
            return ["# Error in supervision action creation"]

    async def _update_supervision_metrics(self, supervision_result: Dict[str, Any]):
        """Update supervision performance metrics"""
        try:
            self.supervision_metrics['tasks_supervised'] += 1
            
            if supervision_result.get('success', False):
                self.supervision_metrics['successful_coordinations'] += 1
            
            # Update coordination efficiency
            efficiency = supervision_result.get('coordination_efficiency', 0.0)
            current_efficiency = self.supervision_metrics['coordination_efficiency']
            tasks_count = self.supervision_metrics['tasks_supervised']
            
            self.supervision_metrics['coordination_efficiency'] = (
                (current_efficiency * (tasks_count - 1) + efficiency) / tasks_count
            )
            
            # Update other metrics
            self.supervision_metrics['agents_managed'] = len(self.managed_agents)
            
        except Exception as e:
            self.logger.error(f"Error updating supervision metrics: {e}")

    async def register_agent(self, agent: BaseQAAgent) -> bool:
        """Register an agent for supervision"""
        try:
            self.managed_agents[agent.agent_id] = agent
            self.agent_workloads[agent.agent_id] = 0.0
            self.agent_capabilities[agent.agent_id] = agent.get_capabilities()
            
            self.logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering agent: {e}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from supervision"""
        try:
            if agent_id in self.managed_agents:
                del self.managed_agents[agent_id]
                del self.agent_workloads[agent_id]
                del self.agent_capabilities[agent_id]
                
                self.logger.info(f"Unregistered agent: {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error unregistering agent: {e}")
            return False

    async def _initialize_system_monitoring(self):
        """Initialize system monitoring capabilities"""
        try:
            # Set up monitoring intervals and thresholds
            self.monitoring_config = {
                'health_check_interval': 30.0,  # seconds
                'performance_analysis_interval': 60.0,
                'alert_thresholds': self.alert_thresholds
            }
            
            # Start monitoring tasks
            asyncio.create_task(self._continuous_system_monitoring())
            
        except Exception as e:
            self.logger.error(f"Error initializing system monitoring: {e}")

    async def _continuous_system_monitoring(self):
        """Continuous system health monitoring"""
        try:
            while self.is_running:
                await asyncio.sleep(self.monitoring_config['health_check_interval'])
                
                # Check system health
                health_score = await self._assess_system_health()
                
                # Check for alerts
                await self._check_system_alerts(health_score)
                
                # Update performance trends
                await self._update_performance_trends()
                
        except asyncio.CancelledError:
            self.logger.info("System monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous system monitoring: {e}")

    async def _check_system_alerts(self, health_score: float):
        """Check for system alerts and issues"""
        try:
            alerts = []
            
            # Check overall health
            if health_score < 0.6:
                alerts.append(f"System health critical: {health_score:.2f}")
            
            # Check agent response times
            for agent_id, agent in self.managed_agents.items():
                if hasattr(agent, 'metrics') and agent.metrics:
                    avg_response_time = agent.metrics.average_response_time
                    if avg_response_time > self.alert_thresholds['agent_response_time']:
                        alerts.append(f"Agent {agent_id} slow response: {avg_response_time:.2f}s")
            
            # Check resource utilization
            avg_workload = sum(self.agent_workloads.values()) / max(1, len(self.agent_workloads))
            if avg_workload > self.alert_thresholds['resource_utilization']:
                alerts.append(f"High resource utilization: {avg_workload:.2f}")
            
            # Log alerts
            for alert in alerts:
                self.logger.warning(f"SYSTEM ALERT: {alert}")
            
        except Exception as e:
            self.logger.error(f"Error checking system alerts: {e}")

    async def _update_performance_trends(self):
        """Update system performance trends"""
        try:
            trend_data = {
                'timestamp': time.time(),
                'system_health': await self._assess_system_health(),
                'coordination_efficiency': self.supervision_metrics['coordination_efficiency'],
                'active_agents': len([a for a in self.managed_agents.values() if a.is_running]),
                'total_tasks': self.supervision_metrics['tasks_supervised'],
                'success_rate': (
                    self.supervision_metrics['successful_coordinations'] /
                    max(1, self.supervision_metrics['tasks_supervised'])
                )
            }
            
            self.performance_trends.append(trend_data)
            
            # Keep only recent trends
            if len(self.performance_trends) > 100:
                self.performance_trends = self.performance_trends[-100:]
            
        except Exception as e:
            self.logger.error(f"Error updating performance trends: {e}")

    async def _load_coordination_patterns(self):
        """Load existing coordination patterns"""
        try:
            # In a full implementation, this would load from persistent storage
            self.coordination_patterns = {
                'sequential_patterns': [],
                'collaborative_patterns': [],
                'pipeline_patterns': []
            }
            
        except Exception as e:
            self.logger.debug(f"No existing coordination patterns to load: {e}")

    async def _save_coordination_patterns(self):
        """Save coordination patterns"""
        try:
            # In a full implementation, this would save to persistent storage
            pass
            
        except Exception as e:
            self.logger.error(f"Error saving coordination patterns: {e}")

    def get_supervision_analytics(self) -> Dict[str, Any]:
        """Get comprehensive supervision analytics"""
        try:
            base_analytics = super().get_execution_analytics()
            
            # Add supervisor-specific analytics
            supervisor_analytics = {
                'supervision_metrics': self.supervision_metrics.copy(),
                'system_state': self.system_state.value,
                'supervision_mode': self.supervision_mode.value,
                'managed_agents_count': len(self.managed_agents),
                'active_tasks_count': len(self.active_tasks),
                'coordination_patterns_count': sum(len(patterns) for patterns in self.coordination_patterns.values()),
                'system_health_score': asyncio.create_task(self._assess_system_health()),
                'performance_trends_count': len(self.performance_trends),
                
                # Agent statistics
                'agent_statistics': {
                    agent_id: {
                        'type': agent.agent_type.value,
                        'running': agent.is_running,
                        'workload': self.agent_workloads.get(agent_id, 0.0),
                        'capabilities_count': len(self.agent_capabilities.get(agent_id, []))
                    }
                    for agent_id, agent in self.managed_agents.items()
                }
            }
            
            # Merge with base analytics
            base_analytics.update(supervisor_analytics)
            
            return base_analytics
            
        except Exception as e:
            self.logger.error(f"Error generating supervision analytics: {e}")
            return {'error': str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'supervisor_agent_id': self.agent_id,
                'system_state': self.system_state.value,
                'supervision_mode': self.supervision_mode.value,
                'managed_agents': {
                    agent_id: {
                        'type': agent.agent_type.value,
                        'status': agent.state.value,
                        'is_running': agent.is_running,
                        'workload': self.agent_workloads.get(agent_id, 0.0)
                    }
                    for agent_id, agent in self.managed_agents.items()
                },
                'active_tasks': len(self.active_tasks),
                'supervision_metrics': self.supervision_metrics.copy(),
                'recent_performance': self.performance_trends[-5:] if self.performance_trends else [],
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


__all__ = [
    "SupervisorAgent",
    "SupervisionTask",
    "AgentCoordination",
    "SystemHealthStatus",
    "SupervisionMode",
    "SystemState"
]