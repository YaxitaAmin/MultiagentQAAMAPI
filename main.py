"""
Main Entry Point for AMAPI System
Advanced Multi-Agent Performance Intelligence System
"""

import asyncio
import signal
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Core imports
from core.logger import AMAPILogger, get_global_logger
from core.attention_economics import AttentionEconomicsEngine
from core.behavioral_learning import BehavioralPatternEngine
from core.device_abstraction import UniversalDeviceAbstraction
from core.llm_interface import LLMInterface

# Agent imports
from agents.supervisor_agent import SupervisorAgent
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent

# Integration imports
from android_env_integration import EnhancedAndroidEnvIntegration, EnhancedQATaskResult

# System imports
from metrics import SystemMetrics
from evaluator import SystemEvaluator
from system_evaluator import EnhancedSystemEvaluator


class AMAPISystem:
    """
    Advanced Multi-Agent Performance Intelligence System
    Main orchestrator for the complete AMAPI ecosystem
    """
    
    def __init__(self, config_path: str = "config/system_config.json"):
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # System state
        self.is_running = False
        self.start_time = None
        self.shutdown_requested = False
        
        # Initialize logger
        self.logger = AMAPILogger("AMAPI_System", self.config.get('logging', {}))
        
        # Core engines
        self.attention_engine: Optional[AttentionEconomicsEngine] = None
        self.behavioral_engine: Optional[BehavioralPatternEngine] = None
        self.device_abstraction: Optional[UniversalDeviceAbstraction] = None
        self.llm_interface: Optional[LLMInterface] = None
        
        # Agents
        self.supervisor_agent: Optional[SupervisorAgent] = None
        self.planner_agent: Optional[PlannerAgent] = None
        self.executor_agent: Optional[ExecutorAgent] = None
        self.verifier_agent: Optional[VerifierAgent] = None
        
        # Integration
        self.android_integration: Optional[EnhancedAndroidEnvIntegration] = None
        
        # System monitoring
        self.metrics: Optional[SystemMetrics] = None
        self.evaluator: Optional[SystemEvaluator] = None
        self.enhanced_evaluator: Optional[EnhancedSystemEvaluator] = None
        
        # System statistics
        self.system_stats = {
            'total_tasks_processed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_uptime': 0.0,
            'average_task_time': 0.0,
            'system_health_score': 1.0,
            'agents_initialized': 0,
            'integrations_active': 0
        }
        
        self.logger.info("AMAPI System initialized")

    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found: {self.config_path}, using defaults")
                return self._get_default_configuration()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}, using defaults")
            return self._get_default_configuration()

    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            "system": {
                "name": "AMAPI",
                "version": "1.0.0",
                "environment": "development",
                "max_concurrent_tasks": 10,
                "task_timeout": 300,
                "health_check_interval": 30
            },
            "logging": {
                "level": "INFO",
                "log_directory": "logs",
                "max_entries": 10000,
                "buffer_size": 100
            },
            "attention": {
                "base_capacity": 10.0,
                "recharge_rate": 1.0,
                "efficiency_decay": 0.1,
                "collaboration_bonus": 0.2
            },
            "behavioral": {
                "min_confidence": 0.7,
                "decay_rate": 0.05,
                "learning_rate": 0.1
            },
            "device_abstraction": {
                "reference_resolution": [1080, 1920],
                "reference_density": 420
            },
            "llm": {
                "default_model": "claude-3-5-sonnet-20241022",
                "anthropic": {
                    "api_key": "your_anthropic_key_here"
                },
                "openai": {
                    "api_key": "your_openai_key_here"
                }
            },
            "android_integration": {
                "task_name": "enhanced_qa_task",
                "emulator_name": "AndroidWorldAvd",
                "screenshots_dir": "screenshots",
                "max_steps": 50
            },
            "agents": {
                "supervisor": {
                    "supervision_mode": "autonomous",
                    "max_managed_agents": 10
                },
                "planner": {
                    "default_strategy": "adaptive",
                    "max_plan_complexity": "highly_complex"
                },
                "executor": {
                    "execution_timeout": 60,
                    "retry_attempts": 3
                },
                "verifier": {
                    "verification_depth": "comprehensive",
                    "confidence_threshold": 0.8
                }
            }
        }

    async def initialize(self) -> bool:
        """Initialize the complete AMAPI system"""
        try:
            self.logger.info("🚀 Starting AMAPI System initialization...")
            
            # Initialize core engines
            await self._initialize_core_engines()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Initialize integrations
            await self._initialize_integrations()
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Perform system health check
            health_status = await self._perform_initial_health_check()
            
            if health_status['overall_health']:
                self.is_running = True
                self.start_time = time.time()
                self.logger.info("✅ AMAPI System initialization completed successfully")
                return True
            else:
                self.logger.error("❌ AMAPI System failed initial health check")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ AMAPI System initialization failed: {e}")
            return False

    async def _initialize_core_engines(self):
        """Initialize core AMAPI engines"""
        try:
            self.logger.info("Initializing core engines...")
            
            # Attention Economics Engine
            self.attention_engine = AttentionEconomicsEngine(self.config.get('attention', {}))
            await self.attention_engine.start_engine()
            self.logger.info("✅ Attention Economics Engine initialized")
            
            # Behavioral Learning Engine
            self.behavioral_engine = BehavioralPatternEngine(self.config.get('behavioral', {}))
            await self.behavioral_engine.start_learning_engine()
            self.logger.info("✅ Behavioral Learning Engine initialized")
            
            # Device Abstraction Layer
            self.device_abstraction = UniversalDeviceAbstraction(self.config.get('device_abstraction', {}))
            self.logger.info("✅ Device Abstraction Layer initialized")
            
            # LLM Interface
            self.llm_interface = LLMInterface(self.config.get('llm', {}))
            await self.llm_interface.initialize()
            self.logger.info("✅ LLM Interface initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing core engines: {e}")
            raise

    async def _initialize_agents(self):
        """Initialize all AMAPI agents"""
        try:
            self.logger.info("Initializing agents...")
            
            # Supervisor Agent
            supervisor_config = {
                **self.config.get('agents', {}).get('supervisor', {}),
                'attention_engine': self.attention_engine,
                'behavioral_engine': self.behavioral_engine,
                'llm_interface': self.llm_interface
            }
            self.supervisor_agent = SupervisorAgent(supervisor_config)
            await self.supervisor_agent.initialize()
            self.system_stats['agents_initialized'] += 1
            self.logger.info("✅ Supervisor Agent initialized")
            
            # Planner Agent
            planner_config = {
                **self.config.get('agents', {}).get('planner', {}),
                'attention_engine': self.attention_engine,
                'behavioral_engine': self.behavioral_engine,
                'llm_interface': self.llm_interface
            }
            self.planner_agent = PlannerAgent(planner_config)
            await self.planner_agent.initialize()
            self.system_stats['agents_initialized'] += 1
            self.logger.info("✅ Planner Agent initialized")
            
            # Executor Agent
            executor_config = {
                **self.config.get('agents', {}).get('executor', {}),
                'attention_engine': self.attention_engine,
                'behavioral_engine': self.behavioral_engine,
                'device_abstraction': self.device_abstraction,
                'llm_interface': self.llm_interface
            }
            self.executor_agent = ExecutorAgent(executor_config)
            await self.executor_agent.initialize()
            self.system_stats['agents_initialized'] += 1
            self.logger.info("✅ Executor Agent initialized")
            
            # Verifier Agent
            verifier_config = {
                **self.config.get('agents', {}).get('verifier', {}),
                'attention_engine': self.attention_engine,
                'behavioral_engine': self.behavioral_engine,
                'llm_interface': self.llm_interface
            }
            self.verifier_agent = VerifierAgent(verifier_config)
            await self.verifier_agent.initialize()
            self.system_stats['agents_initialized'] += 1
            self.logger.info("✅ Verifier Agent initialized")
            
            # Register agents with supervisor
            await self.supervisor_agent.register_agent(self.planner_agent)
            await self.supervisor_agent.register_agent(self.executor_agent)
            await self.supervisor_agent.register_agent(self.verifier_agent)
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise

    async def _initialize_integrations(self):
        """Initialize system integrations"""
        try:
            self.logger.info("Initializing integrations...")
            
            # Android Environment Integration
            android_config = {
                **self.config.get('android_integration', {}),
                'attention_engine': self.attention_engine,
                'device_abstraction': self.device_abstraction,
                'behavioral_engine': self.behavioral_engine
            }
            
            self.android_integration = EnhancedAndroidEnvIntegration(
                config=android_config
            )
            self.system_stats['integrations_active'] += 1
            self.logger.info("✅ Android Environment Integration initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing integrations: {e}")
            raise

    async def _initialize_monitoring(self):
        """Initialize system monitoring and evaluation"""
        try:
            self.logger.info("Initializing monitoring systems...")
            
            # System Metrics
            self.metrics = SystemMetrics(self.config.get('metrics', {}))
            await self.metrics.initialize()
            self.logger.info("✅ System Metrics initialized")
            
            # System Evaluator
            evaluator_config = {
                'attention_engine': self.attention_engine,
                'behavioral_engine': self.behavioral_engine,
                'device_abstraction': self.device_abstraction
            }
            self.evaluator = SystemEvaluator(evaluator_config)
            self.logger.info("✅ System Evaluator initialized")
            
            # Enhanced System Evaluator
            self.enhanced_evaluator = EnhancedSystemEvaluator(evaluator_config)
            await self.enhanced_evaluator.initialize()
            self.logger.info("✅ Enhanced System Evaluator initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing monitoring: {e}")
            raise

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info("✅ Signal handlers configured")
        except Exception as e:
            self.logger.warning(f"Could not setup signal handlers: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    async def _perform_initial_health_check(self) -> Dict[str, Any]:
        """Perform initial system health check"""
        try:
            health_status = {
                'overall_health': True,
                'components': {},
                'issues': [],
                'recommendations': []
            }
            
            # Check core engines
            if self.attention_engine:
                engine_health = self.attention_engine.get_engine_analytics()
                health_status['components']['attention_engine'] = 'healthy'
            else:
                health_status['overall_health'] = False
                health_status['issues'].append('Attention engine not initialized')
            
            if self.behavioral_engine:
                learning_health = self.behavioral_engine.get_learning_analytics()
                health_status['components']['behavioral_engine'] = 'healthy'
            else:
                health_status['overall_health'] = False
                health_status['issues'].append('Behavioral engine not initialized')
            
            # Check agents
            agents_healthy = 0
            total_agents = 4  # supervisor, planner, executor, verifier
            
            for agent_name, agent in [
                ('supervisor', self.supervisor_agent),
                ('planner', self.planner_agent),
                ('executor', self.executor_agent),
                ('verifier', self.verifier_agent)
            ]:
                if agent and agent.is_running:
                    health_status['components'][f'{agent_name}_agent'] = 'healthy'
                    agents_healthy += 1
                else:
                    health_status['components'][f'{agent_name}_agent'] = 'unhealthy'
                    health_status['issues'].append(f'{agent_name} agent not running')
            
            # Check LLM interface
            if self.llm_interface:
                llm_health = await self.llm_interface.health_check()
                if llm_health['overall_health']:
                    health_status['components']['llm_interface'] = 'healthy'
                else:
                    health_status['components']['llm_interface'] = 'degraded'
                    health_status['recommendations'].extend(llm_health.get('recommendations', []))
            
            # Overall health assessment
            if agents_healthy < total_agents * 0.75:  # Less than 75% of agents healthy
                health_status['overall_health'] = False
                health_status['issues'].append('Insufficient healthy agents')
            
            self.system_stats['system_health_score'] = agents_healthy / total_agents
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error performing health check: {e}")
            return {
                'overall_health': False,
                'error': str(e),
                'components': {},
                'issues': ['Health check failed'],
                'recommendations': ['Investigate system state']
            }

    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complete task using the AMAPI system"""
        if not self.is_running:
            return {'success': False, 'error': 'System not running'}
        
        task_start_time = time.time()
        task_id = f"task_{int(task_start_time)}_{hash(task_description) % 10000}"
        
        try:
            self.logger.info(f"🎯 Executing task: {task_description}")
            
            # Update task statistics
            self.system_stats['total_tasks_processed'] += 1
            
            # Create task context
            task_context = {
                'task_id': task_id,
                'description': task_description,
                'start_time': task_start_time,
                'system_config': self.config,
                **(context or {})
            }
            
            # Execute task through supervisor
            if self.supervisor_agent:
                supervision_result, actions = await self.supervisor_agent.predict(
                    task_description,
                    observation={'system_state': 'ready'},
                    context=task_context
                )
                
                task_result = {
                    'success': supervision_result.get('success', True),
                    'task_id': task_id,
                    'description': task_description,
                    'execution_time': time.time() - task_start_time,
                    'supervision_result': supervision_result,
                    'actions_taken': actions,
                    'system_metrics': await self._get_current_system_metrics()
                }
            else:
                # Fallback: execute directly with Android integration
                if self.android_integration:
                    android_result = await self.android_integration.execute_enhanced_qa_task(
                        task_description, max_steps=self.config.get('android_integration', {}).get('max_steps', 50)
                    )
                    
                    task_result = {
                        'success': android_result.success,
                        'task_id': task_id,
                        'description': task_description,
                        'execution_time': android_result.execution_time,
                        'android_result': android_result,
                        'system_metrics': await self._get_current_system_metrics()
                    }
                else:
                    task_result = {
                        'success': False,
                        'error': 'No execution method available',
                        'task_id': task_id
                    }
            
            # Update statistics
            if task_result['success']:
                self.system_stats['successful_tasks'] += 1
            else:
                self.system_stats['failed_tasks'] += 1
            
            # Update average task time
            total_tasks = self.system_stats['total_tasks_processed']
            current_avg = self.system_stats['average_task_time']
            execution_time = task_result['execution_time']
            
            self.system_stats['average_task_time'] = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
            
            # Log task completion
            if task_result['success']:
                self.logger.info(f"✅ Task completed successfully: {task_id} ({execution_time:.2f}s)")
            else:
                self.logger.warning(f"❌ Task failed: {task_id} ({execution_time:.2f}s)")
            
            return task_result
            
        except Exception as e:
            self.system_stats['failed_tasks'] += 1
            self.logger.error(f"❌ Task execution error: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id,
                'execution_time': time.time() - task_start_time
            }

    async def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            metrics = {
                'system_stats': self.system_stats.copy(),
                'uptime': time.time() - self.start_time if self.start_time else 0.0,
                'timestamp': time.time()
            }
            
            # Add engine metrics
            if self.attention_engine:
                metrics['attention_analytics'] = self.attention_engine.get_engine_analytics()
            
            if self.behavioral_engine:
                metrics['behavioral_analytics'] = self.behavioral_engine.get_learning_analytics()
            
            if self.device_abstraction:
                metrics['device_analytics'] = self.device_abstraction.get_abstraction_analytics()
            
            # Add agent metrics
            if self.supervisor_agent:
                metrics['supervisor_analytics'] = self.supervisor_agent.get_supervision_analytics()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'system_info': {
                    'name': self.config.get('system', {}).get('name', 'AMAPI'),
                    'version': self.config.get('system', {}).get('version', '1.0.0'),
                    'environment': self.config.get('system', {}).get('environment', 'development'),
                    'is_running': self.is_running,
                    'uptime': time.time() - self.start_time if self.start_time else 0.0,
                    'shutdown_requested': self.shutdown_requested
                },
                'system_statistics': self.system_stats.copy(),
                'components': {
                    'attention_engine': self.attention_engine is not None,
                    'behavioral_engine': self.behavioral_engine is not None,
                    'device_abstraction': self.device_abstraction is not None,
                    'llm_interface': self.llm_interface is not None,
                    'supervisor_agent': self.supervisor_agent is not None and self.supervisor_agent.is_running,
                    'planner_agent': self.planner_agent is not None and self.planner_agent.is_running,
                    'executor_agent': self.executor_agent is not None and self.executor_agent.is_running,
                    'verifier_agent': self.verifier_agent is not None and self.verifier_agent.is_running,
                    'android_integration': self.android_integration is not None
                },
                'health_metrics': await self._get_current_system_metrics(),
                'timestamp': time.time()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

    async def shutdown(self):
        """Gracefully shutdown the AMAPI system"""
        try:
            self.logger.info("🛑 Initiating AMAPI System shutdown...")
            
            self.is_running = False
            
            # Shutdown agents
            if self.supervisor_agent:
                await self.supervisor_agent.cleanup()
                self.logger.info("✅ Supervisor Agent shutdown")
            
            if self.planner_agent:
                await self.planner_agent.cleanup()
                self.logger.info("✅ Planner Agent shutdown")
            
            if self.executor_agent:
                await self.executor_agent.cleanup()
                self.logger.info("✅ Executor Agent shutdown")
            
            if self.verifier_agent:
                await self.verifier_agent.cleanup()
                self.logger.info("✅ Verifier Agent shutdown")
            
            # Shutdown integrations
            if self.android_integration:
                await self.android_integration.close()
                self.logger.info("✅ Android Integration shutdown")
            
            # Shutdown core engines
            if self.attention_engine:
                await self.attention_engine.stop_engine()
                self.logger.info("✅ Attention Engine shutdown")
            
            if self.behavioral_engine:
                await self.behavioral_engine.stop_learning_engine()
                self.logger.info("✅ Behavioral Engine shutdown")
            
            if self.llm_interface:
                await self.llm_interface.cleanup()
                self.logger.info("✅ LLM Interface shutdown")
            
            # Shutdown monitoring
            if self.metrics:
                await self.metrics.cleanup()
                self.logger.info("✅ Metrics shutdown")
            
            # Final statistics
            if self.start_time:
                total_uptime = time.time() - self.start_time
                self.system_stats['total_uptime'] = total_uptime
                self.logger.info(f"📊 Total uptime: {total_uptime:.2f} seconds")
                self.logger.info(f"📊 Tasks processed: {self.system_stats['total_tasks_processed']}")
                self.logger.info(f"📊 Success rate: {self.system_stats['successful_tasks']}/{self.system_stats['total_tasks_processed']}")
            
            self.logger.info("✅ AMAPI System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def run(self):
        """Main run loop for the AMAPI system"""
        try:
            self.logger.info("🏃 Starting AMAPI System main loop...")
            
            while self.is_running and not self.shutdown_requested:
                # Perform periodic health checks
                await asyncio.sleep(self.config.get('system', {}).get('health_check_interval', 30))
                
                # Update system health
                if self.enhanced_evaluator:
                    health_assessment = await self.enhanced_evaluator.evaluate_system_health()
                    self.system_stats['system_health_score'] = health_assessment.get('overall_score', 1.0)
                
                # Log periodic status
                self.logger.debug(f"System running - Health: {self.system_stats['system_health_score']:.2f}")
            
            # Shutdown requested or system stopped
            await self.shutdown()
            
        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
            await self.shutdown()

    async def run_interactive_mode(self):
        """Run AMAPI in interactive mode for testing"""
        try:
            self.logger.info("🎮 Starting AMAPI Interactive Mode...")
            print("\n=== AMAPI Interactive Mode ===")
            print("Type 'help' for commands, 'quit' to exit")
            
            while self.is_running and not self.shutdown_requested:
                try:
                    user_input = input("\nAMAPI> ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    elif user_input.lower() == 'help':
                        self._print_help()
                    elif user_input.lower() == 'status':
                        status = await self.get_system_status()
                        self._print_status(status)
                    elif user_input.lower() == 'health':
                        health = await self._perform_initial_health_check()
                        self._print_health(health)
                    elif user_input.lower().startswith('exec '):
                        task = user_input[5:]  # Remove 'exec '
                        result = await self.execute_task(task)
                        self._print_task_result(result)
                    elif user_input:
                        # Default: execute as task
                        result = await self.execute_task(user_input)
                        self._print_task_result(result)
                
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except EOFError:
                    print("\nEOF received")
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            print("\nExiting interactive mode...")
            
        except Exception as e:
            self.logger.error(f"Error in interactive mode: {e}")

    def _print_help(self):
        """Print help information"""
        print("\nAvailable commands:")
        print("  help     - Show this help message")
        print("  status   - Show system status")
        print("  health   - Show system health")
        print("  exec <task> - Execute a specific task")
        print("  <task>   - Execute a task (shorthand)")
        print("  quit     - Exit AMAPI")
        print("\nExample tasks:")
        print("  'Open settings app'")
        print("  'Take a screenshot'")
        print("  'Navigate to WiFi settings'")

    def _print_status(self, status: Dict[str, Any]):
        """Print system status"""
        print(f"\n=== System Status ===")
        print(f"Running: {status['system_info']['is_running']}")
        print(f"Uptime: {status['system_info']['uptime']:.1f}s")
        print(f"Tasks processed: {status['system_statistics']['total_tasks_processed']}")
        print(f"Success rate: {status['system_statistics']['successful_tasks']}/{status['system_statistics']['total_tasks_processed']}")
        print(f"Health score: {status['system_statistics']['system_health_score']:.2f}")

    def _print_health(self, health: Dict[str, Any]):
        """Print system health"""
        print(f"\n=== System Health ===")
        print(f"Overall health: {'✅ Healthy' if health['overall_health'] else '❌ Unhealthy'}")
        
        if health.get('components'):
            print("Components:")
            for component, status in health['components'].items():
                emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
                print(f"  {emoji} {component}: {status}")
        
        if health.get('issues'):
            print("Issues:")
            for issue in health['issues']:
                print(f"  ❌ {issue}")

    def _print_task_result(self, result: Dict[str, Any]):
        """Print task execution result"""
        success_emoji = "✅" if result['success'] else "❌"
        print(f"\n{success_emoji} Task Result:")
        print(f"  Success: {result['success']}")
        print(f"  Duration: {result.get('execution_time', 0):.2f}s")
        
        if result.get('error'):
            print(f"  Error: {result['error']}")
        
        if result.get('actions_taken'):
            print(f"  Actions: {len(result['actions_taken'])}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AMAPI - Advanced Multi-Agent Performance Intelligence")
    parser.add_argument("--config", default="config/system_config.json", help="Configuration file path")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--task", help="Execute a single task and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    # Initialize AMAPI system
    amapi = AMAPISystem(args.config)
    
    # Initialize system
    if not await amapi.initialize():
        print("❌ Failed to initialize AMAPI system")
        sys.exit(1)
    
    try:
        if args.task:
            # Execute single task
            print(f"🎯 Executing task: {args.task}")
            result = await amapi.execute_task(args.task)
            
            if result['success']:
                print(f"✅ Task completed successfully in {result['execution_time']:.2f}s")
                sys.exit(0)
            else:
                print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.interactive:
            # Interactive mode
            await amapi.run_interactive_mode()
        
        elif args.daemon:
            # Daemon mode
            await amapi.run()
        
        else:
            # Default: interactive mode
            await amapi.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Ensure cleanup
        if amapi.is_running:
            await amapi.shutdown()


if __name__ == "__main__":
    asyncio.run(main())