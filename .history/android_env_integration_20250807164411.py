"""
AndroidWorld Environment Integration with Agent-S Framework
Enhanced with Universal Device Abstraction and AMAPI Integration
"""

import time
import json
import os
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from loguru import logger

# Enhanced imports for AMAPI integration
from android_env_integration import UniversalDeviceAbstraction, DeviceFingerprint, AndroidVersion
from core.attention_economics import AttentionEconomicsEngine


# AndroidWorld and Agent-S imports
try:
    from android_env.environment import AndroidEnv
    ANDROID_ENV_AVAILABLE = True
    logger.info("AndroidWorld environment imported successfully")
except ImportError as e:
    ANDROID_ENV_AVAILABLE = False
    logger.warning(f"AndroidWorld not available: {e}")


try:
    from gui_agents.s2.agents.agent_s import AgentS2
    from gui_agents.s2.agents.grounding import OSWorldACI
    AGENT_S_AVAILABLE = True
    logger.info("Agent-S framework imported successfully")
except ImportError as e:
    AGENT_S_AVAILABLE = False
    logger.warning(f"Agent-S not available: {e}")


@dataclass
class EnhancedQATaskResult:
    """Enhanced results from QA task execution with AMAPI analytics"""
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
    performance_predictions: Dict[str, Any] = None


@dataclass
class EnhancedAgentAction:
    """Enhanced agent action with attention economics and device adaptation"""
    action_type: str
    parameters: Dict[str, Any]
    timestamp: float
    agent_id: str
    confidence: float = 0.8
    reasoning: Optional[str] = None
    
    # AMAPI enhancements
    attention_cost: float = 0.0
    attention_allocation: Dict[str, float] = None
    device_adaptation: Dict[str, Any] = None
    universal_compatibility: float = 1.0


class EnhancedAndroidEnvIntegration:
    """
    Enhanced AndroidWorld integration with AMAPI, Universal Device Abstraction,
    and Attention Economics
    """
    
    def __init__(self, task_name: str = "settings_wifi", 
                 emulator_name: str = "AndroidWorldAvd",
                 screenshots_dir: str = "screenshots",
                 config: Dict[str, Any] = None):
        """Initialize enhanced AndroidWorld integration"""
        self.task_name = task_name
        self.emulator_name = emulator_name
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        self.config = config or {}
        
        # Core environment state
        self.env = None
        self.agent_s2 = None
        self.grounding_agent = None
        self.current_observation = None
        self.step_count = 0
        self.max_steps = 50
        
        # AMAPI Integration Components
        self.universal_abstraction = UniversalDeviceAbstraction(self.config.get('universal_abstraction', {}))
        self.attention_engine = AttentionEconomicsEngine(self.config.get('attention_economics', {}))
        self.current_device_fingerprint = None
        
        # Enhanced performance tracking
        self.execution_metrics = {
            'total_steps': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'average_step_time': 0.0,
            'task_completion_rate': 0.0,
            
            # AMAPI metrics
            'attention_efficiency': 0.0,
            'device_adaptations': 0,
            'universal_translations': 0,
            'compatibility_score': 1.0,
            'attention_cost_total': 0.0,
            'attention_cost_per_action': 0.0
        }
        
        # Attention tracking
        self.attention_history = []
        self.adaptation_history = []
        
        logger.info(f"Enhanced AndroidEnvIntegration initialized for task: {task_name}")
        
        # Initialize components
        self._setup_enhanced_android_environment()
        self._setup_agent_s_integration()
        self._setup_amapi_integration()
    
    def _setup_enhanced_android_environment(self) -> bool:
        """Setup AndroidWorld environment with device detection"""
        if not ANDROID_ENV_AVAILABLE:
            logger.error("AndroidWorld not available - cannot proceed with real environment")
            return False

        try:
            # Verify emulator is running
            if not self._check_emulator_status():
                logger.error("Android emulator not running - please start emulator first")
                return False

            # Get device information for fingerprinting
            device_info = self._get_device_info()
            
            # Create device fingerprint
            self.current_device_fingerprint = await self.universal_abstraction.detect_device_fingerprint(device_info)
            logger.info(f"Device fingerprinted: {self.current_device_fingerprint.manufacturer} {self.current_device_fingerprint.model}")

            # Initialize AndroidWorld environment
            logger.info("Initializing AndroidWorld environment...")
            self.env = AndroidEnv()
            
            # Reset environment
            try:
                self.env.reset()
                logger.info("AndroidWorld environment initialized successfully")
                return True
            except Exception as reset_error:
                logger.warning(f"Reset with task failed: {reset_error}")
                self.env.reset()
                return True

        except Exception as e:
            logger.error(f"Failed to setup enhanced AndroidWorld environment: {e}")
            return False

    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information from ADB"""
        try:
            device_info = {
                'device_id': 'emulator-5554',  # Default emulator
                'manufacturer': 'Google',
                'model': 'Android Emulator',
                'api_level': 29,
                'screen_width': 1080,
                'screen_height': 1920,
                'screen_density': 420
            }
            
            try:
                # Get actual device properties
                result = subprocess.run(['adb', 'shell', 'getprop', 'ro.product.manufacturer'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    device_info['manufacturer'] = result.stdout.strip()
                
                result = subprocess.run(['adb', 'shell', 'getprop', 'ro.product.model'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    device_info['model'] = result.stdout.strip()
                
                result = subprocess.run(['adb', 'shell', 'getprop', 'ro.build.version.sdk'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    device_info['api_level'] = int(result.stdout.strip())
                
            except Exception as e:
                logger.debug(f"Could not get detailed device info: {e}")
            
            return device_info
            
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {
                'device_id': 'unknown',
                'manufacturer': 'Unknown',
                'model': 'Unknown',
                'api_level': 29
            }
    
    def _setup_agent_s_integration(self) -> bool:
        """Setup Agent-S integration with attention awareness"""
        if not AGENT_S_AVAILABLE:
            logger.error("Agent-S not available - cannot proceed with Agent-S integration")
            return False
        
        try:
            # Setup Agent-S engine parameters
            engine_params = {
                "engine_type": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
            }
            
            # Setup grounding parameters with device-specific adaptations
            engine_params_for_grounding = {
                "engine_type": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "grounding_width": self.current_device_fingerprint.screen_width if self.current_device_fingerprint else 1366,
                "grounding_height": self.current_device_fingerprint.screen_height if self.current_device_fingerprint else 768
            }
            
            # Initialize grounding agent
            self.grounding_agent = OSWorldACI(
                platform="android",  # Changed to android
                engine_params_for_generation=engine_params,
                engine_params_for_grounding=engine_params_for_grounding
            )
            
            # Initialize Agent-S2 with Android platform
            self.agent_s2 = AgentS2(
                engine_params,
                self.grounding_agent,
                platform="android",  # Changed to android
                action_space="android_actions",  # Changed to android actions
                observation_type="screenshot",
                search_engine="Perplexica",
                embedding_engine_type="openai"
            )
            
            logger.info("Enhanced Agent-S integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Agent-S integration: {e}")
            return False
    
    async def _setup_amapi_integration(self) -> bool:
        """Setup AMAPI integration components"""
        try:
            # Initialize attention economics with device-specific parameters
            if self.current_device_fingerprint:
                attention_config = {
                    'device_performance_class': self.current_device_fingerprint.performance_class,
                    'android_version': self.current_device_fingerprint.android_version.value,
                    'screen_density': self.current_device_fingerprint.screen_density
                }
                
                # Initialize attention pools based on device capabilities
                await self.attention_engine.initialize_attention_pools(attention_config)
            
            logger.info("AMAPI integration setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up AMAPI integration: {e}")
            return False
    
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
            logger.warning(f"Could not check emulator status: {e}")
        return False
    
    async def execute_enhanced_qa_task(self, goal: str, max_steps: Optional[int] = None) -> EnhancedQATaskResult:
        """Execute enhanced QA task with full AMAPI integration"""
        start_time = time.time()
        self.max_steps = max_steps or self.max_steps
        self.step_count = 0
        
        agent_logs = []
        screenshots = []
        attention_history = []
        adaptation_history = []
        
        logger.info(f"Starting enhanced QA task execution: {goal}")
        
        try:
            # Initialize attention allocation for task
            task_attention_allocation = await self.attention_engine.allocate_attention_for_task({
                'task_goal': goal,
                'task_complexity': self._estimate_task_complexity(goal),
                'device_performance': self.current_device_fingerprint.performance_class if self.current_device_fingerprint else 'medium'
            })
            
            # Reset environment
            observation = self.reset_environment()
            if observation is None:
                return EnhancedQATaskResult(
                    task_name=self.task_name,
                    success=False,
                    steps_taken=0,
                    execution_time=time.time() - start_time,
                    error_message="Failed to reset environment"
                )
            
            # Main execution loop with AMAPI enhancements
            while self.step_count < self.max_steps:
                step_start_time = time.time()
                
                # Take screenshot with device adaptation
                screenshot_path = await self._save_enhanced_screenshot()
                if screenshot_path:
                    screenshots.append(screenshot_path)
                
                # Get enhanced agent action with attention economics
                action_result = await self._get_enhanced_agent_action(goal, observation, task_attention_allocation)
                if action_result is None:
                    break
                
                # Apply universal device adaptations
                adapted_action = await self._apply_universal_adaptations(action_result)
                
                # Execute enhanced action
                observation, reward, done, info = await self.enhanced_step(adapted_action)
                
                # Track attention usage
                attention_used = action_result.attention_cost
                attention_history.append({
                    'step': self.step_count,
                    'attention_cost': attention_used,
                    'attention_allocation': action_result.attention_allocation,
                    'efficiency': info.get('attention_efficiency', 0.5)
                })
                
                # Track adaptations
                if action_result.device_adaptation:
                    adaptation_history.append({
                        'step': self.step_count,
                        'adaptations': action_result.device_adaptation,
                        'compatibility_score': action_result.universal_compatibility
                    })
                
                # Log enhanced step results
                step_log = {
                    'step': self.step_count,
                    'action': action_result.__dict__,
                    'reward': reward,
                    'done': done,
                    'info': info,
                    'step_time': time.time() - step_start_time,
                    'screenshot': screenshot_path,
                    'attention_cost': attention_used,
                    'device_adaptations': len(action_result.device_adaptation) if action_result.device_adaptation else 0
                }
                agent_logs.append(step_log)
                
                if done:
                    logger.info(f"Enhanced task completed successfully in {self.step_count} steps")
                    break
                    
                self.step_count += 1
            
            # Calculate enhanced metrics
            success = info.get('success', False) if info else False
            execution_time = time.time() - start_time
            
            # Generate attention analytics
            attention_analytics = self._generate_attention_analytics(attention_history)
            
            # Generate device compatibility analytics
            device_compatibility = self._generate_device_compatibility_analytics(adaptation_history)
            
            # Generate performance predictions
            performance_predictions = await self._generate_performance_predictions(agent_logs, attention_analytics)
            
            # Update enhanced metrics
            await self._update_enhanced_execution_metrics(success, execution_time, attention_analytics, device_compatibility)
            
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
                universal_adaptations=adaptation_history,
                performance_predictions=performance_predictions
            )
            
        except Exception as e:
            logger.error(f"Error during enhanced QA task execution: {e}")
            return EnhancedQATaskResult(
                task_name=self.task_name,
                success=False,
                steps_taken=self.step_count,
                execution_time=time.time() - start_time,
                error_message=str(e),
                agent_logs=agent_logs,
                screenshots=screenshots
            )
    
    def reset_environment(self) -> Optional[Dict[str, Any]]:
        """Reset the AndroidWorld environment"""
        try:
            if self.env:
                observation = self.env.reset()
                self.current_observation = observation
                self.step_count = 0
                logger.info("Enhanced environment reset successfully")
                return observation
            else:
                logger.error("Environment not initialized")
                return None
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            return None
    
    async def enhanced_step(self, action: EnhancedAgentAction) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute enhanced step with AMAPI integration"""
        try:
            if not self.env:
                raise ValueError("Environment not initialized")
            
            # Convert Enhanced AgentAction to AndroidWorld action format
            androidworld_action = await self._convert_enhanced_action_format(action)
            
            # Execute action with attention tracking
            step_start_time = time.time()
            observation, reward, done, info = self.env.step(androidworld_action)
            step_time = time.time() - step_start_time
            
            self.current_observation = observation
            
            # Enhanced info with AMAPI data
            enhanced_info = info.copy() if info else {}
            enhanced_info.update({
                'attention_cost': action.attention_cost,
                'attention_efficiency': self._calculate_attention_efficiency(action.attention_cost, step_time, reward),
                'device_adaptations_applied': len(action.device_adaptation) if action.device_adaptation else 0,
                'universal_compatibility': action.universal_compatibility,
                'step_time': step_time
            })
            
            # Update attention economics
            await self.attention_engine.update_attention_usage(
                agent_id='agent_s2',
                attention_cost=action.attention_cost,
                success=enhanced_info.get('success', reward > 0)
            )
            
            # Update metrics
            if enhanced_info.get('success', False):
                self.execution_metrics['successful_actions'] += 1
            else:
                self.execution_metrics['failed_actions'] += 1
            
            self.execution_metrics['total_steps'] += 1
            self.execution_metrics['attention_cost_total'] += action.attention_cost
            
            return observation, reward, done, enhanced_info
            
        except Exception as e:
            logger.error(f"Error during enhanced environment step: {e}")
            return self.current_observation, 0.0, True, {'error': str(e)}
    
    async def _get_enhanced_agent_action(self, goal: str, observation: Dict[str, Any], 
                                       task_attention_allocation: Dict[str, float]) -> Optional[EnhancedAgentAction]:
        """Get enhanced action from Agent-S with attention economics"""
        try:
            if not self.agent_s2:
                logger.error("Agent-S not initialized")
                return None
            
            # Prepare observation for Agent-S
            agent_s_obs = await self._prepare_enhanced_observation_for_agent_s(observation)
            
            # Calculate attention cost for this action
            action_attention_cost = await self.attention_engine.calculate_attention_cost({
                'action_type': 'agent_reasoning',
                'complexity': self._estimate_action_complexity(goal, observation),
                'device_performance': self.current_device_fingerprint.performance_class if self.current_device_fingerprint else 'medium'
            })
            
            # Get action from Agent-S
            info, action = self.agent_s2.predict(
                instruction=goal,
                observation=agent_s_obs
            )
            
            # Create enhanced agent action
            enhanced_action = EnhancedAgentAction(
                action_type=action[0] if isinstance(action, list) else str(action),
                parameters={'raw_action': action},
                timestamp=time.time(),
                agent_id='agent_s2',
                confidence=info.get('confidence', 0.8),
                reasoning=info.get('reasoning', ''),
                attention_cost=action_attention_cost,
                attention_allocation=task_attention_allocation.copy(),
                device_adaptation={},
                universal_compatibility=1.0
            )
            
            return enhanced_action
            
        except Exception as e:
            logger.error(f"Error getting enhanced agent action: {e}")
            return None
    
    async def _apply_universal_adaptations(self, action: EnhancedAgentAction) -> EnhancedAgentAction:
        """Apply universal device adaptations to action"""
        try:
            if not self.current_device_fingerprint:
                return action
            
            # Convert action to universal format
            universal_action = {
                'type': action.action_type,
                'parameters': action.parameters
            }
            
            # Apply universal translation
            translation = await self.universal_abstraction.translate_action_universally(
                universal_action, self.current_device_fingerprint
            )
            
            # Update action with adaptations
            action.device_adaptation = {
                'adaptations_applied': translation.adaptation_notes,
                'compatibility_score': translation.compatibility_score,
                'success_probability': translation.success_probability
            }
            action.universal_compatibility = translation.compatibility_score
            action.parameters = translation.translated_action
            
            # Update metrics
            self.execution_metrics['universal_translations'] += 1
            if len(translation.adaptation_notes) > 0:
                self.execution_metrics['device_adaptations'] += 1
            
            return action
            
        except Exception as e:
            logger.error(f"Error applying universal adaptations: {e}")
            return action
    
    async def _prepare_enhanced_observation_for_agent_s(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare enhanced AndroidWorld observation for Agent-S"""
        try:
            # Extract screenshot
            screenshot = observation.get('screenshot')
            if screenshot is None:
                # Take screenshot using adb as fallback
                screenshot = await self._take_adb_screenshot()
            
            # Enhanced observation with device context
            enhanced_obs = {
                'screenshot': screenshot,
                'ui_elements': observation.get('ui_elements', []),
                'activity': observation.get('activity', ''),
                'timestamp': time.time(),
                
                # Device context
                'device_info': {
                    'manufacturer': self.current_device_fingerprint.manufacturer if self.current_device_fingerprint else 'Unknown',
                    'model': self.current_device_fingerprint.model if self.current_device_fingerprint else 'Unknown',
                    'android_version': self.current_device_fingerprint.android_version.value if self.current_device_fingerprint else 'android_10',
                    'performance_class': self.current_device_fingerprint.performance_class if self.current_device_fingerprint else 'medium'
                },
                
                # Attention context
                'attention_state': await self.attention_engine.get_current_attention_state('agent_s2')
            }
            
            return enhanced_obs
            
        except Exception as e:
            logger.error(f"Error preparing enhanced observation for Agent-S: {e}")
            return {'screenshot': None}
    
    async def _take_adb_screenshot(self) -> Optional[bytes]:
        """Take screenshot using ADB"""
        try:
            result = subprocess.run([
                'adb', 'exec-out', 'screencap', '-p'
            ], capture_output=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.warning("ADB screenshot failed")
                return None
                
        except Exception as e:
            logger.debug(f"Error taking ADB screenshot: {e}")
            return None
    
    async def _convert_enhanced_action_format(self, action: EnhancedAgentAction) -> Dict[str, Any]:
        """Convert Enhanced AgentAction to AndroidWorld action format"""
        try:
            # Extract raw action from Agent-S
            raw_action = action.parameters.get('raw_action')
            
            if isinstance(raw_action, str):
                # Parse string action for Android
                return {'action_type': 'android_execute', 'command': raw_action}
            elif isinstance(raw_action, dict):
                return raw_action
            elif isinstance(raw_action, list) and len(raw_action) > 0:
                # Agent-S2 typically returns list of actions
                return {'action_type': 'android_action', 'action': raw_action[0]}
            else:
                # Default action
                return {'action_type': 'wait', 'duration': 1.0}
                
        except Exception as e:
            logger.error(f"Error converting enhanced action format: {e}")
            return {'action_type': 'wait', 'duration': 1.0}
    
    async def _save_enhanced_screenshot(self) -> Optional[str]:
        """Save enhanced screenshot with device info"""
        try:
            timestamp = int(time.time())
            device_info = f"{self.current_device_fingerprint.manufacturer}_{self.current_device_fingerprint.model}" if self.current_device_fingerprint else "unknown"
            filename = f"step_{self.step_count:03d}_{device_info}_{timestamp}.png"
            filepath = self.screenshots_dir / filename
            
            if self.current_observation and 'screenshot' in self.current_observation:
                screenshot = self.current_observation['screenshot']
                if hasattr(screenshot, 'save'):
                    screenshot.save(str(filepath))
                    return str(filepath)
            
            # Fallback to ADB screenshot
            screenshot_data = await self._take_adb_screenshot()
            if screenshot_data:
                with open(filepath, 'wb') as f:
                    f.write(screenshot_data)
                return str(filepath)
            
            return None
            
        except Exception as e:
            logger.error(f"Error saving enhanced screenshot: {e}")
            return None
    
    def _estimate_task_complexity(self, goal: str) -> float:
        """Estimate task complexity for attention allocation"""
        goal_lower = goal.lower()
        
        complexity_indicators = {
            'simple': ['tap', 'click', 'open', 'close'],
            'medium': ['navigate', 'scroll', 'swipe', 'find'],
            'complex': ['configure', 'setup', 'install', 'uninstall'],
            'very_complex': ['debug', 'troubleshoot', 'analyze', 'compare']
        }
        
        complexity_scores = {'simple': 0.2, 'medium': 0.5, 'complex': 0.8, 'very_complex': 1.0}
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in goal_lower for indicator in indicators):
                return complexity_scores[complexity]
        
        return 0.5  # Default medium complexity
    
    def _estimate_action_complexity(self, goal: str, observation: Dict[str, Any]) -> float:
        """Estimate individual action complexity"""
        # Base complexity from goal
        base_complexity = self._estimate_task_complexity(goal)
        
        # Adjust based on UI elements
        ui_elements = observation.get('ui_elements', [])
        ui_complexity = min(1.0, len(ui_elements) / 20.0)  # Normalize to 0-1
        
        return (base_complexity + ui_complexity) / 2.0
    
    def _calculate_attention_efficiency(self, attention_cost: float, step_time: float, reward: float) -> float:
        """Calculate attention efficiency for this step"""
        try:
            # Efficiency based on reward per attention unit and time
            if attention_cost > 0:
                reward_efficiency = max(0, reward) / attention_cost
                time_efficiency = 1.0 / max(0.1, step_time)  # Faster is better
                return (reward_efficiency + time_efficiency) / 2.0
            else:
                return 0.5  # Default efficiency
        except Exception as e:
            logger.debug(f"Error calculating attention efficiency: {e}")
            return 0.5
    
    def _generate_attention_analytics(self, attention_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive attention analytics"""
        try:
            if not attention_history:
                return {}
            
            total_attention = sum(step['attention_cost'] for step in attention_history)
            avg_attention_per_step = total_attention / len(attention_history)
            
            efficiencies = [step['efficiency'] for step in attention_history]
            avg_efficiency = np.mean(efficiencies) if efficiencies else 0.5
            
            return {
                'total_attention_used': total_attention,
                'average_attention_per_step': avg_attention_per_step,
                'attention_efficiency': avg_efficiency,
                'attention_distribution': self._analyze_attention_distribution(attention_history),
                'efficiency_trend': self._calculate_efficiency_trend(efficiencies),
                'attention_optimization_score': self._calculate_attention_optimization_score(attention_history)
            }
            
        except Exception as e:
            logger.error(f"Error generating attention analytics: {e}")
            return {}
    
    def _analyze_attention_distribution(self, attention_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze how attention was distributed across different allocation pools"""
        try:
            distribution = {}
            
            for step in attention_history:
                allocation = step.get('attention_allocation', {})
                for pool, amount in allocation.items():
                    distribution[pool] = distribution.get(pool, 0) + amount
            
            # Normalize to percentages
            total = sum(distribution.values())
            if total > 0:
                distribution = {pool: (amount / total) * 100 for pool, amount in distribution.items()}
            
            return distribution
            
        except Exception as e:
            logger.debug(f"Error analyzing attention distribution: {e}")
            return {}
    
    def _calculate_efficiency_trend(self, efficiencies: List[float]) -> str:
        """Calculate efficiency trend over time"""
        try:
            if len(efficiencies) < 2:
                return 'stable'
            
            # Simple linear trend
            x = np.arange(len(efficiencies))
            y = np.array(efficiencies)
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.05:
                return 'improving'
            elif slope < -0.05:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.debug(f"Error calculating efficiency trend: {e}")
            return 'unknown'
    
    def _calculate_attention_optimization_score(self, attention_history: List[Dict[str, Any]]) -> float:
        """Calculate how well attention was optimized"""
        try:
            if not attention_history:
                return 0.5
            
            # Score based on efficiency and cost optimization
            efficiencies = [step['efficiency'] for step in attention_history]
            costs = [step['attention_cost'] for step in attention_history]
            
            avg_efficiency = np.mean(efficiencies)
            cost_variance = np.var(costs)  # Lower variance = more consistent
            
            # Combine efficiency and consistency
            optimization_score = avg_efficiency * (1.0 - min(1.0, cost_variance))
            
            return max(0.0, min(1.0, optimization_score))
            
        except Exception as e:
            logger.debug(f"Error calculating attention optimization score: {e}")
            return 0.5
    
    def _generate_device_compatibility_analytics(self, adaptation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate device compatibility analytics"""
        try:
            if not adaptation_history:
                return {'compatibility_score': 1.0, 'adaptations_needed': 0}
            
            compatibility_scores = [step['compatibility_score'] for step in adaptation_history]
            avg_compatibility = np.mean(compatibility_scores)
            
            total_adaptations = sum(len(step['adaptations']) for step in adaptation_history)
            
            return {
                'average_compatibility_score': avg_compatibility,
                'total_adaptations_applied': total_adaptations,
                'adaptations_per_step': total_adaptations / len(adaptation_history),
                'compatibility_trend': self._calculate_compatibility_trend(compatibility_scores),
                'most_common_adaptations': self._find_common_adaptations(adaptation_history)
            }
            
        except Exception as e:
            logger.error(f"Error generating device compatibility analytics: {e}")
            return {}
    
    def _calculate_compatibility_trend(self, compatibility_scores: List[float]) -> str:
        """Calculate compatibility trend over time"""
        try:
            if len(compatibility_scores) < 2:
                return 'stable'
            
            x = np.arange(len(compatibility_scores))
            y = np.array(compatibility_scores)
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.05:
                return 'improving'
            elif slope < -0.05:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.debug(f"Error calculating compatibility trend: {e}")
            return 'unknown'
    
    def _find_common_adaptations(self, adaptation_history: List[Dict[str, Any]]) -> List[str]:
        """Find most commonly applied adaptations"""
        try:
            adaptation_counts = {}
            
            for step in adaptation_history:
                adaptations = step.get('adaptations', {}).get('adaptations_applied', [])
                for adaptation in adaptations:
                    adaptation_counts[adaptation] = adaptation_counts.get(adaptation, 0) + 1
            
            # Sort by frequency
            sorted_adaptations = sorted(adaptation_counts.items(), key=lambda x: x[1], reverse=True)
            
            return [adaptation for adaptation, count in sorted_adaptations[:5]]  # Top 5
            
        except Exception as e:
            logger.debug(f"Error finding common adaptations: {e}")
            return []
    
    async def _generate_performance_predictions(self, agent_logs: List[Dict[str, Any]], 
                                             attention_analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance predictions for future tasks"""
        try:
            if not agent_logs:
                return {}
            
            # Analyze current performance
            success_rate = sum(1 for log in agent_logs if log.get('reward', 0) > 0) / len(agent_logs)
            avg_step_time = np.mean([log['step_time'] for log in agent_logs])
            
            # Predict future performance based on trends
            predicted_improvements = {
                'attention_efficiency': min(1.0, attention_analytics.get('attention_efficiency', 0.5) * 1.1),
                'execution_speed': max(0.1, avg_step_time * 0.9),  # 10% improvement
                'success_probability': min(1.0, success_rate * 1.05)  # 5% improvement
            }
            
            return {
                'current_performance': {
                    'success_rate': success_rate,
                    'average_step_time': avg_step_time,
                    'attention_efficiency': attention_analytics.get('attention_efficiency', 0.5)
                },
                'predicted_improvements': predicted_improvements,
                'confidence': 0.7  # Conservative confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating performance predictions: {e}")
            return {}
    
    async def _update_enhanced_execution_metrics(self, success: bool, execution_time: float,
                                               attention_analytics: Dict[str, Any],
                                               device_compatibility: Dict[str, Any]):
        """Update enhanced execution metrics"""
        try:
            # Update basic metrics
            self.execution_metrics['task_completion_rate'] = (
                self.execution_metrics['successful_actions'] / 
                max(1, self.execution_metrics['total_steps'])
            )
            
            total_steps = self.execution_metrics['total_steps']
            if total_steps > 0:
                current_avg = self.execution_metrics['average_step_time']
                self.execution_metrics['average_step_time'] = (
                    (current_avg * (total_steps - 1) + execution_time) / total_steps
                )
                
                self.execution_metrics['attention_cost_per_action'] = (
                    self.execution_metrics['attention_cost_total'] / total_steps
                )
            
            # Update AMAPI metrics
            self.execution_metrics['attention_efficiency'] = attention_analytics.get('attention_efficiency', 0.5)
            self.execution_metrics['compatibility_score'] = device_compatibility.get('average_compatibility_score', 1.0)
            
        except Exception as e:
            logger.error(f"Error updating enhanced execution metrics: {e}")
    
    def get_enhanced_system_info(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system information"""
        system_info = {
            'android_env_available': ANDROID_ENV_AVAILABLE,
            'agent_s_available': AGENT_S_AVAILABLE,
            'emulator_running': self._check_emulator_status(),
            'task_name': self.task_name,
            'current_step': self.step_count,
            'max_steps': self.max_steps,
            'execution_metrics': self.execution_metrics.copy(),
            'screenshots_saved': len(list(self.screenshots_dir.glob('*.png'))),
            'environment_initialized': self.env is not None,
            'agent_s2_initialized': self.agent_s2 is not None,
            
            # AMAPI enhancements
            'device_fingerprint': asdict(self.current_device_fingerprint) if self.current_device_fingerprint else None,
            'attention_engine_active': self.attention_engine is not None,
            'universal_abstraction_active': self.universal_abstraction is not None,
            'amapi_integration_status': 'active'
        }
        
        return system_info
    
    async def export_enhanced_execution_log(self, filepath: str) -> bool:
        """Export detailed enhanced execution log"""
        try:
            # Get current attention state
            attention_state = await self.attention_engine.get_current_attention_state('agent_s2') if self.attention_engine else {}
            
            # Get universal abstraction analytics
            abstraction_analytics = self.universal_abstraction.get_abstraction_analytics() if self.universal_abstraction else {}
            
            log_data = {
                'system_info': self.get_enhanced_system_info(),
                'execution_metrics': self.execution_metrics,
                'task_configuration': {
                    'task_name': self.task_name,
                    'emulator_name': self.emulator_name,
                    'max_steps': self.max_steps
                },
                'amapi_data': {
                    'attention_state': attention_state,
                    'device_fingerprint': asdict(self.current_device_fingerprint) if self.current_device_fingerprint else None,
                    'universal_abstraction_analytics': abstraction_analytics
                },
                'timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            logger.info(f"Enhanced execution log exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting enhanced execution log: {e}")
            return False
    
    def close(self):
        """Clean up enhanced resources"""
        try:
            if self.env:
                self.env.close()
                logger.info("AndroidWorld environment closed")
            
            if self.agent_s2:
                # Agent-S cleanup if needed
                pass
                
            if self.attention_engine:
                # Attention engine cleanup if needed
                pass
            
        except Exception as e:
            logger.error(f"Error during enhanced cleanup: {e}")


__all__ = [
    "EnhancedAndroidEnvIntegration",
    "EnhancedQATaskResult",
    "EnhancedAgentAction"
]