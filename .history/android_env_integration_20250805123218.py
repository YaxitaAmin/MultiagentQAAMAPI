"""
Enhanced Android Environment Integration - AMAPI + AndroidWorld + Agent-S
Production-ready integration with attention economics and device abstraction
"""

import time
import json
import os
import io
import uuid
import subprocess
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from loguru import logger

from core.logger import AMAPILogger, LogCategory
from core.attention_economics import AttentionEconomicsEngine, AttentionAllocation
from core.device_abstraction import UniversalDeviceAbstraction, DeviceFingerprint, ActionTranslation
from core.behavioral_learning import BehavioralPatternEngine, LearningType


# AndroidWorld integration
try:
    from android_env.environment import AndroidEnv
    ANDROID_ENV_AVAILABLE = True
    logger.info("✅ AndroidWorld environment imported successfully")
except ImportError as e:
    ANDROID_ENV_AVAILABLE = False
    logger.warning(f"❌ AndroidWorld not available: {e}")
    
    # Create fallback AndroidEnv
    class AndroidEnv:
        def __init__(self, *args, **kwargs):
            self.mock_mode = True
            logger.warning("Using AndroidEnv mock")
        
        def reset(self):
            return {'screenshot': None, 'ui_elements': []}
        
        def step(self, action):
            return {'screenshot': None}, 0.0, False, {'success': False}
        
        def close(self):
            pass


# Agent-S integration  
try:
    from gui_agents.s2.agents.agent_s import AgentS2
    from gui_agents.s2.agents.grounding import OSWorldACI
    import pyautogui
    AGENT_S_AVAILABLE = True
    logger.info("✅ Agent-S framework imported successfully")
except ImportError as e:
    AGENT_S_AVAILABLE = False
    logger.warning(f"❌ Agent-S not available: {e}")


@dataclass
class EnhancedQATaskResult:
    """Enhanced results from QA task execution with AMAPI integration"""
    task_name: str
    success: bool
    steps_taken: int
    execution_time: float
    error_message: Optional[str] = None
    agent_logs: List[Dict[str, Any]] = None
    screenshots: List[str] = None
    final_state: Dict[str, Any] = None
    
    # AMAPI enhancements
    attention_analytics: Dict[str, Any] = None
    device_compatibility: Dict[str, Any] = None
    universal_adaptations: List[Dict[str, Any]] = None
    behavioral_patterns: List[str] = None
    learning_events: List[str] = None
    performance_metrics: Dict[str, Any] = None


@dataclass 
class EnhancedAgentAction:
    """Enhanced agent action with AMAPI integration"""
    action_type: str
    parameters: Dict[str, Any]
    timestamp: float
    agent_id: str
    confidence: float = 0.8
    reasoning: Optional[str] = None
    
    # AMAPI enhancements
    attention_cost: float = 0.0
    device_adaptation: Optional[ActionTranslation] = None
    pattern_match: Optional[str] = None
    learning_context: Dict[str, Any] = None


class EnhancedAndroidEnvIntegration:
    """
    Enhanced AndroidWorld Integration with AMAPI
    Combines AndroidWorld, Agent-S, attention economics, device abstraction, and behavioral learning
    """
    
    def __init__(self, task_name: str = "enhanced_qa_task", 
                 emulator_name: str = "AndroidWorldAvd",
                 screenshots_dir: str = "enhanced_screenshots",
                 config: Dict[str, Any] = None):
        """Initialize enhanced AndroidWorld integration"""
        self.task_name = task_name
        self.emulator_name = emulator_name
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        self.config = config or {}
        
        # Environment state
        self.env: Optional[AndroidEnv] = None
        self.agent_s2: Optional[AgentS2] = None
        self.grounding_agent = None
        self.current_observation = None
        self.step_count = 0
        self.max_steps = 50
        
        # AMAPI Integration
        self.attention_engine = AttentionEconomicsEngine(self.config.get('attention', {}))
        self.device_abstraction = UniversalDeviceAbstraction(self.config.get('device_abstraction', {}))
        self.behavioral_engine = BehavioralPatternEngine(self.config.get('behavioral', {}))
        
        # Device fingerprint
        self.current_device: Optional[DeviceFingerprint] = None
        
        # Enhanced metrics
        self.execution_metrics = {
            'total_steps': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'average_step_time': 0.0,
            'task_completion_rate': 0.0,
            'attention_efficiency': 0.0,
            'device_compatibility_score': 1.0,
            'patterns_applied': 0,
            'adaptations_made': 0,
            'learning_events_generated': 0
        }
        
        # Action history for learning
        self.action_history: List[EnhancedAgentAction] = []
        self.learning_events: List[str] = []
        
        # Logger
        self.logger = AMAPILogger(f"EnhancedAndroidEnv_{task_name}")
        
        self.logger.info(f"Initialized Enhanced AndroidEnv Integration for task: {task_name}")
        
        # Initialize components
        asyncio.create_task(self._initialize_async_components())

    async def _initialize_async_components(self):
        """Initialize async components"""
        try:
            # Setup Android environment
            await self._setup_android_environment()
            
            # Setup Agent-S integration
            await self._setup_agent_s_integration()
            
            # Initialize AMAPI engines
            await self._initialize_amapi_engines()
            
            # Detect device
            await self._detect_and_fingerprint_device()
            
            self.logger.info("Enhanced Android environment fully initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing async components: {e}")

    async def _setup_android_environment(self) -> bool:
        """Setup AndroidWorld environment"""
        if not ANDROID_ENV_AVAILABLE:
            self.logger.error("AndroidWorld not available - using fallback mode")
            self.env = AndroidEnv()  # Fallback mock
            return False

        try:
            # Verify emulator is running
            if not self._check_emulator_status():
                self.logger.warning("Android emulator not running - please start emulator first")
                # Continue with mock for development
                self.env = AndroidEnv()
                return False

            # Initialize AndroidWorld environment
            self.logger.info("Initializing AndroidWorld environment...")
            
            # Initialize AndroidEnv with proper configuration
            self.env = AndroidEnv(
                # Configure AndroidEnv based on your specific setup
                # You may need to add specific parameters here
            )

            # Reset environment
            try:
                initial_observation = self.env.reset()
                self.current_observation = initial_observation
                self.logger.info("AndroidWorld environment initialized and reset successfully")
                return True
                
            except Exception as reset_error:
                self.logger.warning(f"Environment reset failed: {reset_error}")
                # Try basic reset
                try:
                    self.env.reset()
                    return True
                except Exception as e:
                    self.logger.error(f"Basic reset also failed: {e}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to setup AndroidWorld environment: {e}")
            # Use fallback mock
            self.env = AndroidEnv()
            return False

    async def _setup_agent_s_integration(self) -> bool:
        """Setup Agent-S integration"""
        if not AGENT_S_AVAILABLE:
            self.logger.error("Agent-S not available - cannot proceed with Agent-S integration")
            return False
        
        try:
            # Setup Agent-S engine parameters
            engine_params = {
                "engine_type": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
            }
            
            # Setup grounding parameters
            engine_params_for_grounding = {
                "engine_type": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "grounding_width": 1366,
                "grounding_height": 768
            }
            
            # Initialize grounding agent
            self.grounding_agent = OSWorldACI(
                platform="linux",
                engine_params_for_generation=engine_params,
                engine_params_for_grounding=engine_params_for_grounding
            )
            
            # Initialize Agent-S2
            self.agent_s2 = AgentS2(
                engine_params=engine_params,
                grounding_agent=self.grounding_agent,
                platform="linux",
                action_space="pyautogui", 
                observation_type="screenshot",
                search_engine="Perplexica",
                embedding_engine_type="openai"
            )
            
            self.logger.info("Agent-S integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup Agent-S integration: {e}")
            return False

    async def _initialize_amapi_engines(self):
        """Initialize AMAPI engines"""
        try:
            # Start attention economics engine
            await self.attention_engine.start_engine()
            
            # Start behavioral learning engine
            await self.behavioral_engine.start_learning_engine()
            
            # Initialize attention pools for Android environment
            await self.attention_engine.initialize_agent_attention(
                "android_env_agent",
                {
                    'perception': 3.0,  # High for UI analysis
                    'reasoning': 2.5,   # Medium for action planning
                    'execution': 4.0,   # High for Android actions
                    'learning': 1.5,    # Medium for pattern learning
                    'monitoring': 2.0   # Medium for verification
                }
            )
            
            self.logger.info("AMAPI engines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing AMAPI engines: {e}")

    async def _detect_and_fingerprint_device(self):
        """Detect and fingerprint the current Android device"""
        try:
            # Get device information from ADB or environment
            device_info = await self._get_device_info()
            
            # Create device fingerprint
            self.current_device = await self.device_abstraction.detect_device_fingerprint(device_info)
            
            self.logger.info(f"Device fingerprinted: {self.current_device.manufacturer} {self.current_device.model}")
            
        except Exception as e:
            self.logger.error(f"Error detecting device fingerprint: {e}")
            # Create fallback device fingerprint
            self.current_device = await self.device_abstraction.detect_device_fingerprint({
                'manufacturer': 'Generic',
                'model': 'AndroidDevice',
                'api_level': 30,
                'screen_width': 1080,
                'screen_height': 1920,
                'screen_density': 420
            })

    async def _get_device_info(self) -> Dict[str, Any]:
        """Get device information from ADB"""
        try:
            device_info = {}
            
            # Try to get device info from ADB
            if self._check_emulator_status():
                try:
                    # Get device properties
                    props_result = subprocess.run(
                        ['adb', 'shell', 'getprop'],
                        capture_output=True, text=True, timeout=10
                    )
                    
                    if props_result.returncode == 0:
                        props = props_result.stdout
                        
                        # Parse device properties
                        if 'ro.product.manufacturer' in props:
                            for line in props.split('\n'):
                                if 'ro.product.manufacturer' in line:
                                    device_info['manufacturer'] = line.split(': [')[1].split(']')[0]
                                elif 'ro.product.model' in line:
                                    device_info['model'] = line.split(': [')[1].split(']')[0]
                                elif 'ro.build.version.sdk' in line:
                                    device_info['api_level'] = int(line.split(': [')[1].split(']')[0])
                    
                    # Get screen dimensions
                    size_result = subprocess.run(
                        ['adb', 'shell', 'wm', 'size'],
                        capture_output=True, text=True, timeout=5
                    )
                    
                    if size_result.returncode == 0 and 'Physical size:' in size_result.stdout:
                        size_line = size_result.stdout.strip()
                        dimensions = size_line.split('Physical size: ')[1].split('x')
                        device_info['screen_width'] = int(dimensions[0])
                        device_info['screen_height'] = int(dimensions[1])
                    
                    # Get screen density
                    density_result = subprocess.run(
                        ['adb', 'shell', 'wm', 'density'],
                        capture_output=True, text=True, timeout=5
                    )
                    
                    if density_result.returncode == 0 and 'Physical density:' in density_result.stdout:
                        density_line = density_result.stdout.strip()
                        device_info['screen_density'] = int(density_line.split('Physical density: ')[1])
                
                except subprocess.TimeoutExpired:
                    self.logger.warning("ADB commands timed out")
                except Exception as e:
                    self.logger.warning(f"Error getting device info from ADB: {e}")
            
            # Fill in defaults for missing info
            device_info.setdefault('manufacturer', 'Google')
            device_info.setdefault('model', 'AndroidEmulator')
            device_info.setdefault('api_level', 30)
            device_info.setdefault('screen_width', 1080)
            device_info.setdefault('screen_height', 1920)
            device_info.setdefault('screen_density', 420)
            device_info.setdefault('capabilities', ['touch', 'ui_automation'])
            
            return device_info
            
        except Exception as e:
            self.logger.error(f"Error getting device info: {e}")
            return {
                'manufacturer': 'Unknown',
                'model': 'UnknownDevice',
                'api_level': 30,
                'screen_width': 1080,
                'screen_height': 1920,
                'screen_density': 420,
                'capabilities': []
            }

    def _check_emulator_status(self) -> bool:
        """Check if Android emulator is running"""
        try:
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                devices = [line for line in result.stdout.split('\n') 
                          if '\tdevice' in line]
                return len(devices) > 0
        except Exception as e:
            self.logger.warning(f"Could not check emulator status: {e}")
        return False

    async def execute_enhanced_qa_task(self, goal: str, max_steps: Optional[int] = None) -> EnhancedQATaskResult:
        """Execute enhanced QA task with AMAPI integration"""
        start_time = time.time()
        self.max_steps = max_steps or self.max_steps
        self.step_count = 0
        
        agent_logs = []
        screenshots = []
        learning_events = []
        applied_patterns = []
        universal_adaptations = []
        
        self.logger.info(f"Starting enhanced QA task execution: {goal}")
        
        try:
            # Allocate attention for the entire task
            task_attention = await self.attention_engine.allocate_attention_for_task({
                'agent_id': 'android_env_agent',
                'task_description': goal,
                'task_complexity': 0.7,  # Android tasks are generally complex
                'agent_type': 'android_executor'
            })
            
            # Reset environment
            observation = await self._reset_environment_async()
            if observation is None:
                return self._create_failed_result("Failed to reset environment", start_time)
            
            # Main execution loop with AMAPI enhancements
            while self.step_count < self.max_steps:
                step_start_time = time.time()
                
                # Take screenshot
                screenshot_path = await self._save_screenshot_async()
                if screenshot_path:
                    screenshots.append(screenshot_path)
                
                # Get relevant behavioral patterns
                relevant_patterns = await self.behavioral_engine.get_relevant_patterns(
                    'android_env_agent',
                    {'task': goal, 'step': self.step_count, 'observation': observation}
                )
                
                # Get enhanced agent action with AMAPI integration
                enhanced_action = await self._get_enhanced_agent_action(
                    goal, observation, task_attention, relevant_patterns
                )
                
                if enhanced_action is None:
                    break
                
                # Apply universal device adaptation
                if self.current_device and enhanced_action.parameters:
                    action_translation = await self.device_abstraction.translate_action_universally(
                        enhanced_action.parameters, self.current_device
                    )
                    enhanced_action.device_adaptation = action_translation
                    universal_adaptations.append({
                        'step': self.step_count,
                        'original_action': action_translation.original_action,
                        'translated_action': action_translation.translated_action,
                        'adaptations': action_translation.adaptation_notes,
                        'compatibility_score': action_translation.compatibility_score
                    })
                
                # Execute enhanced action
                observation, reward, done, info = await self._step_async(enhanced_action)
                
                # Record learning event
                learning_event_id = await self.behavioral_engine.record_learning_event(
                    'android_env_agent',
                    LearningType.SUCCESS_PATTERN if info.get('success', False) else LearningType.FAILURE_PATTERN,
                    {
                        'action': enhanced_action.parameters,
                        'context': {'goal': goal, 'step': self.step_count},
                        'outcome': 'success' if info.get('success', False) else 'failure',
                        'reward': reward
                    },
                    'success' if info.get('success', False) else 'failure',
                    {'reward': reward, 'step_time': time.time() - step_start_time}
                )
                
                if learning_event_id:
                    learning_events.append(learning_event_id)
                
                # Update attention usage
                await self.attention_engine.update_attention_usage(
                    'android_env_agent',
                    enhanced_action.attention_cost,
                    info.get('success', False),
                    'android_action',
                    task_attention.allocation_id
                )
                
                # Log enhanced step results
                step_log = {
                    'step': self.step_count,
                    'action': asdict(enhanced_action),
                    'reward': reward,
                    'done': done,
                    'info': info,
                    'step_time': time.time() - step_start_time,
                    'screenshot': screenshot_path,
                    'attention_cost': enhanced_action.attention_cost,
                    'device_adaptation': asdict(enhanced_action.device_adaptation) if enhanced_action.device_adaptation else None,
                    'pattern_match': enhanced_action.pattern_match,
                    'learning_event': learning_event_id
                }
                agent_logs.append(step_log)
                
                # Store action in history
                self.action_history.append(enhanced_action)
                
                if done:
                    self.logger.info(f"Enhanced task completed successfully in {self.step_count} steps")
                    break
                    
                self.step_count += 1
            
            # Calculate final metrics
            success = info.get('success', False) if info else False
            execution_time = time.time() - start_time
            
            # Update execution metrics
            await self._update_enhanced_execution_metrics(success, execution_time, universal_adaptations, learning_events)
            
            # Get attention analytics
            attention_analytics = await self._get_attention_analytics(task_attention)
            
            # Get device compatibility analytics
            device_compatibility = await self._get_device_compatibility_analytics()
            
            # Create enhanced result
            return EnhancedQATaskResult(
                task_name=self.task_name,
                success=success,
                steps_taken=self.step_count,
                execution_time=execution_time,
                agent_logs=agent_logs,
                screenshots=screenshots,
                final_state=observation,
                attention_analytics=attention_analytics,
                device_compatibility=device_compatibility,
                universal_adaptations=universal_adaptations,
                behavioral_patterns=applied_patterns,
                learning_events=learning_events,
                performance_metrics=self.execution_metrics.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Error during enhanced QA task execution: {e}")
            return self._create_failed_result(str(e), start_time, agent_logs, screenshots)

    def _create_failed_result(self, error_message: str, start_time: float, 
                            agent_logs: List = None, screenshots: List = None) -> EnhancedQATaskResult:
        """Create failed task result"""
        return EnhancedQATaskResult(
            task_name=self.task_name,
            success=False,
            steps_taken=self.step_count,
            execution_time=time.time() - start_time,
            error_message=error_message,
            agent_logs=agent_logs or [],
            screenshots=screenshots or [],
            attention_analytics={'error': 'Task failed'},
            device_compatibility={'error': 'Task failed'},
            universal_adaptations=[],
            behavioral_patterns=[],
            learning_events=[],
            performance_metrics=self.execution_metrics.copy()
        )

    async def _reset_environment_async(self) -> Optional[Dict[str, Any]]:
        """Reset the AndroidWorld environment asynchronously"""
        try:
            if self.env:
                # Reset in thread pool to avoid blocking
                observation = await asyncio.to_thread(self.env.reset)
                self.current_observation = observation
                self.step_count = 0
                self.logger.info("Environment reset successfully")
                return observation
            else:
                self.logger.error("Environment not initialized")
                return None
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {e}")
            return None

    async def _step_async(self, action: EnhancedAgentAction) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute a single step in the environment asynchronously"""
        try:
            if not self.env:
                raise ValueError("Environment not initialized")
            
            # Convert EnhancedAgentAction to AndroidWorld action format
            androidworld_action = self._convert_enhanced_action_format(action)
            
            # Execute action in thread pool
            result = await asyncio.to_thread(self.env.step, androidworld_action)
            observation, reward, done, info = result
            
            self.current_observation = observation
            
            # Update metrics
            if info.get('success', False):
                self.execution_metrics['successful_actions'] += 1
            else:
                self.execution_metrics['failed_actions'] += 1
            
            self.execution_metrics['total_steps'] += 1
            
            return observation, reward, done, info
            
        except Exception as e:
            self.logger.error(f"Error during environment step: {e}")
            return self.current_observation, 0.0, True, {'error': str(e)}

    async def _get_enhanced_agent_action(self, goal: str, observation: Dict[str, Any],
                                       attention_allocation: AttentionAllocation,
                                       relevant_patterns: List) -> Optional[EnhancedAgentAction]:
        """Get enhanced agent action with AMAPI integration"""
        try:
            action_attention_cost = 0.0
            pattern_match = None
            
            # Check if we can use Agent-S
            if self.agent_s2:
                try:
                    # Prepare observation for Agent-S
                    agent_s_obs = self._prepare_observation_for_agent_s(observation)
                    
                    # Get action from Agent-S
                    info, action = await asyncio.to_thread(
                        self.agent_s2.predict,
                        instruction=goal,
                        observation=agent_s_obs
                    )
                    
                    # Calculate attention cost
                    action_attention_cost = self._calculate_action_attention_cost(action, info)
                    
                    # Check for pattern matches
                    if relevant_patterns:
                        best_pattern = max(relevant_patterns, key=lambda p: p.effectiveness_score)
                        if best_pattern.effectiveness_score > 0.8:
                            pattern_match = best_pattern.pattern_id
                            self.execution_metrics['patterns_applied'] += 1
                    
                    # Create enhanced action
                    enhanced_action = EnhancedAgentAction(
                        action_type=action[0] if isinstance(action, list) else str(action),
                        parameters={'raw_action': action, 'agent_s_info': info},
                        timestamp=time.time(),
                        agent_id='enhanced_agent_s2',
                        confidence=info.get('confidence', 0.8),
                        reasoning=info.get('reasoning', ''),
                        attention_cost=action_attention_cost,
                        pattern_match=pattern_match,
                        learning_context={
                            'goal': goal,
                            'step': self.step_count,
                            'attention_allocation': asdict(attention_allocation)
                        }
                    )
                    
                    return enhanced_action
                    
                except Exception as e:
                    self.logger.error(f"Error getting Agent-S action: {e}")
            
            # Fallback: Generate action based on patterns or simple heuristics
            fallback_action = await self._generate_fallback_action(goal, observation, relevant_patterns)
            fallback_action.attention_cost = 1.0  # Default attention cost
            
            return fallback_action
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced agent action: {e}")
            return None

    async def _generate_fallback_action(self, goal: str, observation: Dict[str, Any],
                                      relevant_patterns: List) -> EnhancedAgentAction:
        """Generate fallback action when Agent-S is not available"""
        try:
            # Simple action generation based on goal keywords
            goal_lower = goal.lower()
            
            if 'tap' in goal_lower or 'click' in goal_lower:
                action_type = 'tap'
                parameters = {'action_type': 'tap', 'coordinates': [540, 960]}  # Center screen
            elif 'swipe' in goal_lower:
                action_type = 'swipe'
                parameters = {'action_type': 'swipe', 'coordinates': [540, 1200, 540, 800]}  # Swipe up
            elif 'type' in goal_lower or 'input' in goal_lower:
                action_type = 'type'
                parameters = {'action_type': 'type', 'text': 'test_input'}
            else:
                action_type = 'wait'
                parameters = {'action_type': 'wait', 'duration': 1.0}
            
            return EnhancedAgentAction(
                action_type=action_type,
                parameters=parameters,
                timestamp=time.time(),
                agent_id='fallback_agent',
                confidence=0.6,
                reasoning=f"Fallback action for goal: {goal}",
                learning_context={'goal': goal, 'step': self.step_count}
            )
            
        except Exception as e:
            self.logger.error(f"Error generating fallback action: {e}")
            return EnhancedAgentAction(
                action_type='wait',
                parameters={'action_type': 'wait', 'duration': 1.0},
                timestamp=time.time(),
                agent_id='error_fallback',
                confidence=0.3,
                reasoning=f"Error fallback: {str(e)}"
            )

    def _calculate_action_attention_cost(self, action: Any, info: Dict[str, Any]) -> float:
        """Calculate attention cost for an action"""
        try:
            base_cost = 1.0
            
            # Adjust based on action complexity
            if isinstance(action, list):
                base_cost += len(action) * 0.2
            
            # Adjust based on confidence
            confidence = info.get('confidence', 0.8)
            if confidence < 0.5:
                base_cost *= 1.5  # More attention needed for uncertain actions
            
            return min(base_cost, 5.0)  # Cap at 5.0
            
        except Exception as e:
            self.logger.error(f"Error calculating attention cost: {e}")
            return 1.0

    def _prepare_observation_for_agent_s(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare AndroidWorld observation for Agent-S"""
        try:
            # Extract screenshot
            screenshot = observation.get('screenshot')
            if screenshot is None:
                # Take screenshot using pyautogui as fallback
                try:
                    import pyautogui
                    screenshot = pyautogui.screenshot()
                    buffered = io.BytesIO()
                    screenshot.save(buffered, format="PNG")
                    screenshot = buffered.getvalue()
                except Exception as e:
                    self.logger.warning(f"Failed to capture fallback screenshot: {e}")
                    screenshot = None
            
            return {
                'screenshot': screenshot,
                'ui_elements': observation.get('ui_elements', []),
                'activity': observation.get('activity', ''),
                'timestamp': time.time(),
                'enhanced_by': 'amapi'
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing observation for Agent-S: {e}")
            return {'screenshot': None, 'error': str(e)}

    def _convert_enhanced_action_format(self, action: EnhancedAgentAction) -> Dict[str, Any]:
        """Convert EnhancedAgentAction to AndroidWorld action format"""
        try:
            # Use device adaptation if available
            if action.device_adaptation:
                translated_action = action.device_adaptation.translated_action
                if translated_action:
                    return translated_action
            
            # Extract raw action from parameters
            raw_action = action.parameters.get('raw_action')
            
            if isinstance(raw_action, str):
                # Parse string action
                return {'action_type': 'execute', 'code': raw_action}
            elif isinstance(raw_action, list) and len(raw_action) > 0:
                # Handle list of actions
                return {'action_type': 'execute', 'code': raw_action[0]}
            elif isinstance(raw_action, dict):
                return raw_action
            else:
                # Default action based on action type
                if action.action_type == 'tap':
                    return {
                        'action_type': 'tap',
                        'coordinates': action.parameters.get('coordinates', [540, 960])
                    }
                elif action.action_type == 'swipe':
                    return {
                        'action_type': 'swipe',
                        'coordinates': action.parameters.get('coordinates', [540, 1200, 540, 800])
                    }
                else:
                    return {'action_type': 'wait', 'duration': 1.0}
                
        except Exception as e:
            self.logger.error(f"Error converting enhanced action format: {e}")
            return {'action_type': 'wait', 'duration': 1.0}

    async def _save_screenshot_async(self) -> Optional[str]:
        """Save current screenshot asynchronously"""
        try:
            timestamp = int(time.time())
            filename = f"enhanced_step_{self.step_count:03d}_{timestamp}.png"
            filepath = self.screenshots_dir / filename
            
            if self.current_observation and 'screenshot' in self.current_observation:
                screenshot = self.current_observation['screenshot']
                if hasattr(screenshot, 'save'):
                    await asyncio.to_thread(screenshot.save, str(filepath))
                    return str(filepath)
            
            # Fallback to pyautogui
            try:
                import pyautogui
                screenshot = await asyncio.to_thread(pyautogui.screenshot)
                await asyncio.to_thread(screenshot.save, str(filepath))
                return str(filepath)
            except Exception as e:
                self.logger.warning(f"Fallback screenshot failed: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error saving screenshot: {e}")
            return None

    async def _update_enhanced_execution_metrics(self, success: bool, execution_time: float,
                                               adaptations: List, learning_events: List):
        """Update enhanced execution metrics"""
        try:
            # Update basic metrics
            self.execution_metrics['task_completion_rate'] = (
                self.execution_metrics['successful_actions'] / 
                max(1, self.execution_metrics['total_steps'])
            )
            
            # Update average step time
            total_steps = self.execution_metrics['total_steps']
            current_avg = self.execution_metrics['average_step_time']
            self.execution_metrics['average_step_time'] = (
                (current_avg * (total_steps - 1) + execution_time) / total_steps
            )
            
            # Update AMAPI-specific metrics
            self.execution_metrics['adaptations_made'] = len(adaptations)
            self.execution_metrics['learning_events_generated'] = len(learning_events)
            
            # Calculate device compatibility score
            if adaptations:
                compatibility_scores = [a.get('compatibility_score', 1.0) for a in adaptations]
                self.execution_metrics['device_compatibility_score'] = sum(compatibility_scores) / len(compatibility_scores)
            
            # Calculate attention efficiency
            if self.action_history:
                attention_costs = [a.attention_cost for a in self.action_history]
                total_attention = sum(attention_costs)
                successful_actions = sum(1 for a in self.action_history if a.confidence > 0.7)
                
                if total_attention > 0:
                    self.execution_metrics['attention_efficiency'] = successful_actions / total_attention
            
        except Exception as e:
            self.logger.error(f"Error updating enhanced execution metrics: {e}")

    async def _get_attention_analytics(self, task_attention: AttentionAllocation) -> Dict[str, Any]:
        """Get attention analytics for the task"""
        try:
            attention_state = await self.attention_engine.get_current_attention_state('android_env_agent')
            engine_analytics = self.attention_engine.get_engine_analytics()
            
            return {
                'task_attention_allocation': asdict(task_attention),
                'current_attention_state': attention_state,
                'attention_efficiency': self.execution_metrics.get('attention_efficiency', 0.0),
                'total_attention_used': sum(a.attention_cost for a in self.action_history),
                'average_attention_per_action': (
                    sum(a.attention_cost for a in self.action_history) / len(self.action_history)
                    if self.action_history else 0.0
                ),
                'engine_analytics': engine_analytics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting attention analytics: {e}")
            return {'error': str(e)}

    async def _get_device_compatibility_analytics(self) -> Dict[str, Any]:
        """Get device compatibility analytics"""
        try:
            if not self.current_device:
                return {'error': 'No device fingerprint available'}
            
            device_profile = self.device_abstraction.get_device_profile(self.current_device.device_id)
            abstraction_analytics = self.device_abstraction.get_abstraction_analytics()
            
            # Calculate adaptation success rate
            adaptations_made = self.execution_metrics.get('adaptations_made', 0)
            successful_actions = self.execution_metrics.get('successful_actions', 0)
            
            adaptation_success_rate = (
                successful_actions / max(1, adaptations_made)
                if adaptations_made > 0 else 1.0
            )
            
            return {
                'device_profile': device_profile,
                'compatibility_score': self.execution_metrics.get('device_compatibility_score', 1.0),
                'adaptations_made': adaptations_made,
                'adaptation_success_rate': adaptation_success_rate,
                'average_compatibility_score': abstraction_analytics.get('average_device_compatibility', 1.0),
                'abstraction_analytics': abstraction_analytics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting device compatibility analytics: {e}")
            return {'error': str(e)}

    def get_enhanced_system_info(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system information"""
        try:
            base_info = {
                'android_env_available': ANDROID_ENV_AVAILABLE,
                'agent_s_available': AGENT_S_AVAILABLE,
                'emulator_running': self._check_emulator_status(),
                'task_name': self.task_name,
                'current_step': self.step_count,
                'max_steps': self.max_steps,
                'screenshots_saved': len(list(self.screenshots_dir.glob('*.png'))),
                'environment_initialized': self.env is not None,
                'agent_s2_initialized': self.agent_s2 is not None
            }
            
            # Add AMAPI enhancements
            enhanced_info = {
                **base_info,
                'attention_engine_active': hasattr(self.attention_engine, '_recharge_task') and self.attention_engine._recharge_task is not None,
                'behavioral_engine_active': hasattr(self.behavioral_engine, '_learning_task') and self.behavioral_engine._learning_task is not None,
                'device_abstraction_active': self.device_abstraction is not None,
                'current_device': asdict(self.current_device) if self.current_device else None,
                'execution_metrics': self.execution_metrics.copy(),
                'action_history_size': len(self.action_history),
                'learning_events_count': len(self.learning_events)
            }
            
            return enhanced_info
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced system info: {e}")
            return {'error': str(e)}

    async def export_enhanced_execution_log(self, filepath: str) -> bool:
        """Export detailed enhanced execution log"""
        try:
            # Get analytics from all engines
            attention_analytics = self.attention_engine.get_engine_analytics()
            behavioral_analytics = self.behavioral_engine.get_learning_analytics()
            device_analytics = self.device_abstraction.get_abstraction_analytics()
            
            log_data = {
                'system_info': self.get_enhanced_system_info(),
                'execution_metrics': self.execution_metrics,
                'task_configuration': {
                    'task_name': self.task_name,
                    'emulator_name': self.emulator_name,
                    'max_steps': self.max_steps
                },
                'amapi_analytics': {
                    'attention_analytics': attention_analytics,
                    'behavioral_analytics': behavioral_analytics,
                    'device_analytics': device_analytics
                },
                'action_history': [asdict(action) for action in self.action_history],
                'learning_events': self.learning_events,
                'timestamp': time.time()
            }
            
            # Save to file
            await asyncio.to_thread(self._write_log_file, filepath, log_data)
            
            self.logger.info(f"Enhanced execution log exported to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting enhanced execution log: {e}")
            return False

    def _write_log_file(self, filepath: str, log_data: Dict[str, Any]):
        """Write log file synchronously"""
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

    async def close(self):
        """Clean up enhanced resources"""
        try:
            # Close AndroidWorld environment
            if self.env:
                await asyncio.to_thread(self.env.close)
                self.logger.info("AndroidWorld environment closed")
            
            # Stop AMAPI engines
            await self.attention_engine.stop_engine()
            await self.behavioral_engine.stop_learning_engine()
            
            # Clear caches and histories
            self.action_history.clear()
            self.learning_events.clear()
            
            self.logger.info("Enhanced Android environment integration closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during enhanced cleanup: {e}")


__all__ = [
    "EnhancedAndroidEnvIntegration",
    "EnhancedQATaskResult", 
    "EnhancedAgentAction"
]