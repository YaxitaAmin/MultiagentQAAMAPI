"""
Executor Agent for AMAPI System
Specializes in action execution with device abstraction and attention optimization
"""

import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from agents.base_agent import BaseQAAgent, AgentType, AgentAction, AgentObservation
from agents.planner_agent import QAPlan
from core.device_abstraction import UniversalDeviceAbstraction, DeviceFingerprint, ActionTranslation
from core.attention_economics import AttentionEconomicsEngine
from android_env_integration import EnhancedAndroidEnvIntegration
# from utils.device_helpers import DeviceCompatibilityChecker
from utils.attention_helpers import calculate_execution_attention_cost


class DeviceType(Enum):
    """Supported device types"""
    EMULATOR = "emulator"
    REAL_DEVICE = "real_device"
    SIMULATOR = "simulator"


@dataclass
class DeviceConfig:
    """Device configuration"""
    device_type: DeviceType
    screen_width: int
    screen_height: int
    density: int
    api_level: int
    manufacturer: str = "Google"
    model: str = "Android Device"
    capabilities: List[str] = None


@dataclass
class ExecutionResult:
    """Result of action execution"""
    execution_id: str
    action: AgentAction
    success: bool
    execution_time: float
    attention_used: float
    device_adaptations: List[str]
    screenshots: List[str]
    error_message: Optional[str] = None
    device_compatibility: float = 1.0
    performance_metrics: Dict[str, Any] = None


class ExecutorAgent(BaseQAAgent):
    """
    Advanced Executor Agent with Universal Device Abstraction
    Executes QA actions with intelligent device adaptation and attention optimization
    """

    def __init__(self, device_config: DeviceConfig, config: Dict[str, Any] = None):
        super().__init__(
            agent_type=AgentType.EXECUTOR,
            config=config
        )
        
        # Device configuration
        self.device_config = device_config
        self.current_device_fingerprint: Optional[DeviceFingerprint] = None
        
        # Execution components
        self.device_abstraction = UniversalDeviceAbstraction(config.get('device_abstraction', {}))
        self.android_env = None
        self.compatibility_checker = DeviceCompatibilityChecker()
        
        # Execution state
        self.active_executions: Dict[str, ExecutionResult] = {}
        self.execution_history: List[ExecutionResult] = []
        self.current_plan: Optional[QAPlan] = None
        
        # Executor specializations
        self.specializations = [
            'action_execution',
            'device_adaptation',
            'ui_automation',
            'error_recovery',
            'performance_optimization'
        ]
        
        # Execution metrics
        self.execution_metrics = {
            'actions_executed': 0,
            'successful_executions': 0,
            'device_adaptations_applied': 0,
            'average_execution_time': 0.0,
            'attention_efficiency': 0.0,
            'compatibility_score': 1.0,
            'error_recovery_rate': 0.0
        }

    async def _initialize_agent_systems(self) -> None:
        """Initialize executor-specific systems"""
        try:
            # Initialize Android environment integration
            await self._initialize_android_environment()
            
            # Detect and fingerprint device
            await self._initialize_device_detection()
            
            # Initialize device abstraction
            await self._initialize_device_abstraction()
            
            # Load execution patterns
            await self._load_execution_patterns()
            
            self.logger.info("Executor agent systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize executor systems: {e}")
            raise

    async def _cleanup_agent_systems(self) -> None:
        """Cleanup executor-specific systems"""
        try:
            if self.android_env:
                self.android_env.close()
            
            await self._save_execution_patterns()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up executor systems: {e}")

    async def _initialize_android_environment(self) -> None:
        """Initialize enhanced Android environment"""
        try:
            env_config = {
                'attention_economics': self.config.get('attention_economics', {}),
                'universal_abstraction': self.config.get('universal_abstraction', {})
            }
            
            self.android_env = EnhancedAndroidEnvIntegration(
                task_name="amapi_execution",
                config=env_config
            )
            
            # The Android environment will be initialized when needed
            self.logger.info("Android environment integration ready")
            
        except Exception as e:
            self.logger.error(f"Error initializing Android environment: {e}")
            raise

    async def _initialize_device_detection(self) -> None:
        """Initialize device detection and fingerprinting"""
        try:
            # Create device info from config
            device_info = {
                'device_id': f"device_{uuid.uuid4().hex[:8]}",
                'manufacturer': self.device_config.manufacturer,
                'model': self.device_config.model,
                'api_level': self.device_config.api_level,
                'screen_width': self.device_config.screen_width,
                'screen_height': self.device_config.screen_height,
                'screen_density': self.device_config.density,
                'capabilities': self.device_config.capabilities or []
            }
            
            # Generate device fingerprint
            self.current_device_fingerprint = await self.device_abstraction.detect_device_fingerprint(device_info)
            
            self.logger.info(f"Device fingerprinted: {self.current_device_fingerprint.manufacturer} {self.current_device_fingerprint.model}")
            
        except Exception as e:
            self.logger.error(f"Error in device detection: {e}")
            raise

    async def _initialize_device_abstraction(self) -> None:
        """Initialize device abstraction layer"""
        try:
            # Set up compatibility matrix for this device
            compatibility_score = await self._calculate_device_compatibility()
            self.execution_metrics['compatibility_score'] = compatibility_score
            
            self.logger.info(f"Device abstraction initialized (compatibility: {compatibility_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error initializing device abstraction: {e}")
            raise

    async def predict(self, instruction: str, observation: Dict[str, Any], 
                     context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], List[AgentAction]]:
        """Execute QA actions with device adaptation and attention optimization"""
        try:
            # Extract execution plan from context
            execution_plan = context.get('execution_plan') if context else None
            if execution_plan:
                self.current_plan = execution_plan
            
            # Allocate attention for execution
            attention_allocation = await self.allocate_attention(
                f"Execute: {instruction}",
                context.get('task_complexity', 0.5) if context else 0.5
            )
            
            # Execute with enhanced Android environment
            execution_result = await self._execute_with_enhanced_android_env(
                instruction, observation, context, attention_allocation
            )
            
            # Create execution actions
            actions = await self._create_execution_actions(execution_result)
            
            # Update execution metrics
            await self._update_execution_metrics(execution_result)
            
            # Create reasoning info
            reasoning_info = {
                'execution_id': execution_result.execution_id,
                'success': execution_result.success,
                'execution_time': execution_result.execution_time,
                'attention_used': execution_result.attention_used,
                'device_adaptations': execution_result.device_adaptations,
                'device_compatibility': execution_result.device_compatibility,
                'screenshots_captured': len(execution_result.screenshots),
                'performance_metrics': execution_result.performance_metrics,
                'error_message': execution_result.error_message,
                'execution_results': [asdict(execution_result)]
            }
            
            # Store active execution
            self.active_executions[execution_result.execution_id] = execution_result
            
            self.logger.info(f"Execution completed: {execution_result.execution_id} (success: {execution_result.success})")
            
            return reasoning_info, actions
            
        except Exception as e:
            self.logger.error(f"Error in executor prediction: {e}")
            return {'error': str(e), 'success': False}, []

    async def _execute_with_enhanced_android_env(self, instruction: str, observation: Dict[str, Any],
                                               context: Dict[str, Any], attention_allocation) -> ExecutionResult:
        """Execute using enhanced Android environment"""
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Prepare execution context
            execution_context = {
                'instruction': instruction,
                'observation': observation,
                'device_fingerprint': asdict(self.current_device_fingerprint) if self.current_device_fingerprint else {},
                'attention_allocation': asdict(attention_allocation),
                'execution_plan': asdict(self.current_plan) if self.current_plan else {}
            }
            
            # Execute task with Android environment
            if not self.android_env:
                raise ValueError("Android environment not initialized")
            
            # Execute enhanced QA task
            task_result = await self.android_env.execute_enhanced_qa_task(
                goal=instruction,
                max_steps=context.get('max_steps', 20) if context else 20
            )
            
            execution_time = time.time() - start_time
            
            # Extract execution information
            success = task_result.success
            attention_used = sum([
                log.get('attention_cost', 0) for log in task_result.agent_logs or []
            ])
            
            device_adaptations = []
            if task_result.universal_adaptations:
                for adaptation in task_result.universal_adaptations:
                    device_adaptations.extend(adaptation.get('adaptations', {}).get('adaptations_applied', []))
            
            screenshots = task_result.screenshots or []
            error_message = task_result.error_message
            
            # Calculate device compatibility
            device_compatibility = (
                task_result.device_compatibility.get('average_compatibility_score', 1.0)
                if task_result.device_compatibility else 1.0
            )
            
            # Create performance metrics
            performance_metrics = {
                'steps_taken': task_result.steps_taken,
                'attention_efficiency': task_result.attention_analytics.get('attention_efficiency', 0.5) if task_result.attention_analytics else 0.5,
                'device_adaptations_count': len(device_adaptations),
                'execution_efficiency': min(1.0, 30.0 / execution_time),  # 30s baseline
                'android_env_metrics': task_result.attention_analytics or {}
            }
            
            # Create execution result
            execution_result = ExecutionResult(
                execution_id=execution_id,
                action=self._create_executed_action(instruction, execution_context),
                success=success,
                execution_time=execution_time,
                attention_used=attention_used,
                device_adaptations=device_adaptations,
                screenshots=screenshots,
                error_message=error_message,
                device_compatibility=device_compatibility,
                performance_metrics=performance_metrics
            )
            
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.logger.error(f"Error in enhanced Android environment execution: {e}")
            
            # Return failed execution result
            return ExecutionResult(
                execution_id=execution_id,
                action=self._create_executed_action(instruction, {}),
                success=False,
                execution_time=execution_time,
                attention_used=1.0,  # Minimal attention for failed execution
                device_adaptations=[],
                screenshots=[],
                error_message=str(e),
                device_compatibility=0.5,
                performance_metrics={'error': True}
            )

    def _create_executed_action(self, instruction: str, execution_context: Dict[str, Any]) -> AgentAction:
        """Create AgentAction representing the executed action"""
        return AgentAction(
            action_id=f"action_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            action_type="android_execution",
            parameters={
                'instruction': instruction,
                'execution_context': execution_context
            },
            timestamp=time.time(),
            confidence=0.8,
            reasoning=f"Executed Android task: {instruction}",
            attention_cost=execution_context.get('attention_allocation', {}).get('total_attention', 2.0),
            expected_outcome="Android task completion"
        )

    async def _create_execution_actions(self, execution_result: ExecutionResult) -> List[AgentAction]:
        """Create execution actions from result"""
        actions = []
        
        try:
            # Main execution action
            main_action = AgentAction(
                action_id=f"exec_action_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                action_type="execution_complete",
                parameters={
                    'execution_id': execution_result.execution_id,
                    'success': execution_result.success,
                    'execution_time': execution_result.execution_time,
                    'device_adaptations': execution_result.device_adaptations,
                    'performance_metrics': execution_result.performance_metrics
                },
                timestamp=time.time(),
                confidence=0.9 if execution_result.success else 0.3,
                reasoning=f"Execution {'completed successfully' if execution_result.success else 'failed'}",
                attention_cost=execution_result.attention_used,
                expected_outcome="Execution result processed"
            )
            actions.append(main_action)
            
            # Device adaptation actions
            if execution_result.device_adaptations:
                adaptation_action = AgentAction(
                    action_id=f"adaptation_{uuid.uuid4().hex[:8]}",
                    agent_id=self.agent_id,
                    action_type="device_adaptation",
                    parameters={
                        'adaptations_applied': execution_result.device_adaptations,
                        'compatibility_score': execution_result.device_compatibility,
                        'device_fingerprint': asdict(self.current_device_fingerprint) if self.current_device_fingerprint else {}
                    },
                    timestamp=time.time(),
                    confidence=execution_result.device_compatibility,
                    reasoning=f"Applied {len(execution_result.device_adaptations)} device adaptations",
                    attention_cost=0.1 * len(execution_result.device_adaptations),
                    expected_outcome="Device compatibility optimized"
                )
                actions.append(adaptation_action)
            
            # Screenshot capture actions
            if execution_result.screenshots:
                screenshot_action = AgentAction(
                    action_id=f"screenshot_{uuid.uuid4().hex[:8]}",
                    agent_id=self.agent_id,
                    action_type="screenshot_capture",
                    parameters={
                        'screenshots_captured': len(execution_result.screenshots),
                        'screenshot_paths': execution_result.screenshots
                    },
                    timestamp=time.time(),
                    confidence=0.9,
                    reasoning=f"Captured {len(execution_result.screenshots)} screenshots",
                    attention_cost=0.05 * len(execution_result.screenshots),
                    expected_outcome="Visual execution record created"
                )
                actions.append(screenshot_action)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error creating execution actions: {e}")
            return []

    async def _update_execution_metrics(self, execution_result: ExecutionResult):
        """Update execution performance metrics"""
        try:
            self.execution_metrics['actions_executed'] += 1
            
            if execution_result.success:
                self.execution_metrics['successful_executions'] += 1
            
            if execution_result.device_adaptations:
                self.execution_metrics['device_adaptations_applied'] += len(execution_result.device_adaptations)
            
            # Update average execution time
            current_avg = self.execution_metrics['average_execution_time']
            actions_count = self.execution_metrics['actions_executed']
            
            self.execution_metrics['average_execution_time'] = (
                (current_avg * (actions_count - 1) + execution_result.execution_time) / actions_count
            )
            
            # Update attention efficiency
            attention_efficiency = execution_result.performance_metrics.get('attention_efficiency', 0.5)
            current_efficiency = self.execution_metrics['attention_efficiency']
            
            self.execution_metrics['attention_efficiency'] = (
                (current_efficiency * (actions_count - 1) + attention_efficiency) / actions_count
            )
            
            # Update compatibility score
            current_compatibility = self.execution_metrics['compatibility_score']
            self.execution_metrics['compatibility_score'] = (
                (current_compatibility * (actions_count - 1) + execution_result.device_compatibility) / actions_count
            )
            
            # Store execution result
            self.execution_history.append(execution_result)
            
            # Keep only recent executions in memory
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
        except Exception as e:
            self.logger.error(f"Error updating execution metrics: {e}")

    async def _calculate_device_compatibility(self) -> float:
        """Calculate compatibility score for current device"""
        try:
            if not self.current_device_fingerprint:
                return 0.5
            
            # Use compatibility checker to assess device
            compatibility_factors = {
                'android_version': self.current_device_fingerprint.android_version.value,
                'performance_class': self.current_device_fingerprint.performance_class,
                'screen_density': self.current_device_fingerprint.screen_density,
                'capabilities': len(self.current_device_fingerprint.capabilities)
            }
            
            # Calculate compatibility score
            compatibility_score = self.compatibility_checker.calculate_compatibility_score(compatibility_factors)
            
            return compatibility_score
            
        except Exception as e:
            self.logger.error(f"Error calculating device compatibility: {e}")
            return 0.5

    async def _load_execution_patterns(self):
        """Load existing execution patterns"""
        try:
            # In a full implementation, this would load from persistent storage
            self.execution_patterns = {
                'successful_actions': {},
                'failed_actions': {},
                'device_adaptations': {},
                'attention_optimizations': {}
            }
            
        except Exception as e:
            self.logger.debug(f"No existing execution patterns to load: {e}")

    async def _save_execution_patterns(self):
        """Save execution patterns"""
        try:
            # In a full implementation, this would save to persistent storage
            pass
            
        except Exception as e:
            self.logger.error(f"Error saving execution patterns: {e}")

    async def execute_plan_step(self, plan: QAPlan, step_number: int) -> ExecutionResult:
        """Execute a specific step from a QA plan"""
        try:
            if step_number > len(plan.execution_steps):
                raise ValueError(f"Step {step_number} not found in plan {plan.plan_id}")
            
            step = plan.execution_steps[step_number - 1]
            step_instruction = f"Execute step {step_number}: {step['action']}"
            
            # Create context for step execution
            context = {
                'plan_id': plan.plan_id,
                'step_number': step_number,
                'step_details': step,
                'max_steps': 5,  # Limited steps for individual plan step
                'task_complexity': plan.complexity_score
            }
            
            # Execute step
            reasoning_info, actions = await self.predict(
                step_instruction, 
                {'plan_step': step}, 
                context
            )
            
            # Return the execution result
            execution_id = reasoning_info.get('execution_id')
            if execution_id and execution_id in self.active_executions:
                return self.active_executions[execution_id]
            else:
                # Create a basic execution result
                return ExecutionResult(
                    execution_id=f"step_{uuid.uuid4().hex[:8]}",
                    action=self._create_executed_action(step_instruction, context),
                    success=reasoning_info.get('success', False),
                    execution_time=reasoning_info.get('execution_time', 0.0),
                    attention_used=reasoning_info.get('attention_used', 1.0),
                    device_adaptations=reasoning_info.get('device_adaptations', []),
                    screenshots=[],
                    device_compatibility=reasoning_info.get('device_compatibility', 1.0),
                    performance_metrics=reasoning_info.get('performance_metrics', {})
                )
            
        except Exception as e:
            self.logger.error(f"Error executing plan step: {e}")
            raise

    async def recover_from_error(self, execution_result: ExecutionResult, 
                               recovery_strategy: str = "retry") -> ExecutionResult:
        """Attempt to recover from execution error"""
        try:
            recovery_id = f"recovery_{uuid.uuid4().hex[:8]}"
            
            if recovery_strategy == "retry":
                # Simple retry with same parameters
                self.logger.info(f"Retrying failed execution: {execution_result.execution_id}")
                
                # Re-execute the original action
                original_instruction = execution_result.action.parameters.get('instruction', 'Unknown task')
                
                recovery_result = await self._execute_with_enhanced_android_env(
                    original_instruction,
                    {},
                    {'max_steps': 10, 'recovery_attempt': True},
                    await self.allocate_attention(f"Recovery: {original_instruction}", 0.6)
                )
                
                recovery_result.execution_id = recovery_id
                
                # Update error recovery rate
                if recovery_result.success:
                    recovery_rate = self.execution_metrics.get('error_recovery_rate', 0.0)
                    total_errors = sum(1 for result in self.execution_history if not result.success)
                    
                    if total_errors > 0:
                        self.execution_metrics['error_recovery_rate'] = min(1.0, recovery_rate + (1.0 / total_errors))
                
                return recovery_result
            
            else:
                # Other recovery strategies could be implemented
                self.logger.warning(f"Unsupported recovery strategy: {recovery_strategy}")
                return execution_result
            
        except Exception as e:
            self.logger.error(f"Error in error recovery: {e}")
            return execution_result

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of specific execution"""
        try:
            if execution_id in self.active_executions:
                result = self.active_executions[execution_id]
                return {
                    'execution_id': execution_id,
                    'status': 'completed',
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'device_adaptations': len(result.device_adaptations),
                    'attention_used': result.attention_used
                }
            
            # Check execution history
            for result in self.execution_history:
                if result.execution_id == execution_id:
                    return {
                        'execution_id': execution_id,
                        'status': 'archived',
                        'success': result.success,
                        'execution_time': result.execution_time,
                        'device_adaptations': len(result.device_adaptations),
                        'attention_used': result.attention_used
                    }
            
            return {'execution_id': execution_id, 'status': 'not_found'}
            
        except Exception as e:
            self.logger.error(f"Error getting execution status: {e}")
            return {'execution_id': execution_id, 'status': 'error', 'error': str(e)}

    def get_device_status(self) -> Dict[str, Any]:
        """Get current device status and capabilities"""
        try:
            device_status = {
                'device_config': asdict(self.device_config),
                'device_fingerprint': asdict(self.current_device_fingerprint) if self.current_device_fingerprint else None,
                'compatibility_score': self.execution_metrics['compatibility_score'],
                'android_env_available': self.android_env is not None,
                'device_abstraction_active': self.device_abstraction is not None
            }
            
            # Add Android environment status if available
            if self.android_env:
                device_status['android_env_status'] = self.android_env.get_enhanced_system_info()
            
            return device_status
            
        except Exception as e:
            self.logger.error(f"Error getting device status: {e}")
            return {'error': str(e)}

    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive execution analytics"""
        try:
            base_analytics = super().get_execution_analytics()
            
            # Add executor-specific analytics
            executor_analytics = {
                'execution_metrics': self.execution_metrics.copy(),
                'device_compatibility': self.execution_metrics['compatibility_score'],
                'active_executions': len(self.active_executions),
                'execution_history_size': len(self.execution_history),
                'device_fingerprint': asdict(self.current_device_fingerprint) if self.current_device_fingerprint else None,
                
                # Performance analytics
                'success_rate': (
                    self.execution_metrics['successful_executions'] / 
                    max(1, self.execution_metrics['actions_executed'])
                ),
                'adaptation_frequency': (
                    self.execution_metrics['device_adaptations_applied'] /
                    max(1, self.execution_metrics['actions_executed'])
                ),
                'attention_efficiency': self.execution_metrics['attention_efficiency']
            }
            
            # Merge with base analytics
            base_analytics.update(executor_analytics)
            
            return base_analytics
            
        except Exception as e:
            self.logger.error(f"Error generating execution analytics: {e}")
            return {'error': str(e)}

    def get_capabilities(self) -> List[str]:
        """Get executor capabilities"""
        base_capabilities = super().get_capabilities()
        
        executor_capabilities = [
            'android_automation',
            'device_adaptation',
            'ui_interaction',
            'screenshot_capture',
            'error_recovery',
            'attention_optimization',
            'performance_monitoring'
        ]
        
        return base_capabilities + executor_capabilities


__all__ = [
    "ExecutorAgent",
    "DeviceConfig",
    "DeviceType",
    "ExecutionResult"
]