"""
Command Line Interface for Multi-Agent QA System
Provides various commands for testing, benchmarking, and system control
"""

import argparse
import asyncio
import sys
import time
import json
from typing import Dict, Any, List
from loguru import logger

from orchestrator import QAOrchestrator
from agents.executor_agent import DeviceConfig, DeviceType
from core.amapi_core import LearningType


class QASystemCLI:
    """
    Command Line Interface for QA System
    Supports various operational modes and benchmarking
    """

    def __init__(self):
        self.orchestrator = None
        self.config = {
            'planner': {'max_plan_complexity': 'high'},
            'executor': {'retry_attempts': 3},
            'verifier': {'strict_verification': True},
            'supervisor': {'intervention_threshold': 0.7},
            'amapi': {'learning_rate': 0.1}
        }

    async def initialize_system(self, device_type: str = "emulator", 
                              device_id: str = None) -> None:
        """Initialize the QA system"""
        logger.info("🚀 Initializing Multi-Agent QA System...")
        
        try:
            # Create device config
            device_config = DeviceConfig(
                device_type=DeviceType(device_type),
                device_id=device_id,
                screen_width=1080,
                screen_height=1920,
                api_level=29
            )
            
            # Initialize orchestrator
            self.orchestrator = QAOrchestrator(self.config)
            await self.orchestrator.initialize_system(device_config)
            
            logger.info("✅ System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise

    async def run_wifi_test(self) -> Dict[str, Any]:
        """Run WiFi toggle test"""
        logger.info("📶 Running WiFi toggle test...")
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        result = await self.orchestrator.execute_wifi_toggle_task()
        execution_time = time.time() - start_time
        
        if result.get('success'):
            logger.info(f"✅ WiFi test completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"❌ WiFi test failed: {result.get('error', 'Unknown error')}")
        
        return result

    async def run_settings_test(self) -> Dict[str, Any]:
        """Run settings navigation test"""
        logger.info("⚙️ Running settings navigation test...")
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        result = await self.orchestrator.execute_settings_navigation_task()
        execution_time = time.time() - start_time
        
        if result.get('success'):
            logger.info(f"✅ Settings test completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"❌ Settings test failed: {result.get('error', 'Unknown error')}")
        
        return result

    async def run_app_test(self, app_name: str) -> Dict[str, Any]:
        """Run app launch test"""
        logger.info(f"📱 Running {app_name} launch test...")
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        result = await self.orchestrator.execute_app_launch_task(app_name)
        execution_time = time.time() - start_time
        
        if result.get('success'):
            logger.info(f"✅ {app_name} test completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"❌ {app_name} test failed: {result.get('error', 'Unknown error')}")
        
        return result

    async def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run comprehensive QA test suite"""
        logger.info("🧪 Running comprehensive QA test suite...")
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        result = await self.orchestrator.execute_comprehensive_qa_suite()
        execution_time = time.time() - start_time
        
        success_rate = result.get('success_rate', 0)
        total_tasks = result.get('total_tasks', 0)
        successful_tasks = result.get('successful_tasks', 0)
        
        logger.info(f"🏁 QA Suite completed in {execution_time:.2f}s")
        logger.info(f"📊 Results: {successful_tasks}/{total_tasks} tasks successful ({success_rate:.1%})")
        
        return result

    async def run_custom_task(self, task_description: str, 
                            priority: int = 5, 
                            expected_duration: float = 60.0) -> Dict[str, Any]:
        """Run custom QA task"""
        logger.info(f"🎯 Running custom task: {task_description}")
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        requirements = {
            'priority': priority,
            'expected_duration': expected_duration
        }
        
        start_time = time.time()
        result = await self.orchestrator.execute_qa_task(task_description, requirements)
        execution_time = time.time() - start_time
        
        if result.get('success'):
            logger.info(f"✅ Custom task completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"❌ Custom task failed: {result.get('error', 'Unknown error')}")
        
        return result

    async def run_benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        """Run performance benchmark"""
        logger.info(f"⚡ Running performance benchmark ({iterations} iterations)...")
        
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        benchmark_start = time.time()
        results = {
            'total_iterations': iterations,
            'successful_iterations': 0,
            'failed_iterations': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'iteration_results': []
        }
        
        # Benchmark tasks
        benchmark_tasks = [
            ("wifi_test", self.run_wifi_test),
            ("settings_test", self.run_settings_test),
            ("camera_test", lambda: self.run_app_test("Camera"))
        ]
        
        for iteration in range(iterations):
            logger.info(f"🔄 Benchmark iteration {iteration + 1}/{iterations}")
            
            iteration_start = time.time()
            iteration_results = []
            iteration_success = True
            
            # Run all benchmark tasks
            for task_name, task_func in benchmark_tasks:
                try:
                    task_result = await task_func()
                    iteration_results.append({
                        'task': task_name,
                        'success': task_result.get('success', False),
                        'execution_time': task_result.get('execution_time', 0)
                    })
                    
                    if not task_result.get('success', False):
                        iteration_success = False
                        
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    iteration_results.append({
                        'task': task_name,
                        'success': False,
                        'error': str(e)
                    })
                    iteration_success = False
            
            iteration_time = time.time() - iteration_start
            
            # Update results
            if iteration_success:
                results['successful_iterations'] += 1
            else:
                results['failed_iterations'] += 1
            
            results['min_time'] = min(results['min_time'], iteration_time)
            results['max_time'] = max(results['max_time'], iteration_time)
            results['iteration_results'].append({
                'iteration': iteration + 1,
                'success': iteration_success,
                'time': iteration_time,
                'tasks': iteration_results
            })
        
        # Finalize results
        results['total_time'] = time.time() - benchmark_start
        results['average_time'] = results['total_time'] / iterations
        results['success_rate'] = results['successful_iterations'] / iterations
        
        # Fix min_time if no successful iterations
        if results['min_time'] == float('inf'):
            results['min_time'] = 0.0
        
        logger.info(f"📊 Benchmark completed:")
        logger.info(f"  • Success Rate: {results['success_rate']:.1%}")
        logger.info(f"  • Average Time: {results['average_time']:.2f}s")
        logger.info(f"  • Min Time: {results['min_time']:.2f}s")
        logger.info(f"  • Max Time: {results['max_time']:.2f}s")
        
        return results

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.orchestrator:
            return {'error': 'System not initialized'}
        
        return self.orchestrator.get_system_status()

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        if not self.orchestrator:
            return {'error': 'System not initialized'}
        
        return self.orchestrator.get_performance_report()

    async def get_amapi_analytics(self) -> Dict[str, Any]:
        """Get AMAPI learning analytics"""
        if not self.orchestrator or not self.orchestrator.amapi_core:
            return {'error': 'AMAPI not available'}
        
        return self.orchestrator.amapi_core.get_amapi_analytics()

    def print_system_status(self, status: Dict[str, Any]) -> None:
        """Print formatted system status"""
        print("\n" + "=" * 60)
        print("🤖 AMAPI QA System Status")
        print("=" * 60)
        
        if 'error' in status:
            print(f"❌ Error: {status['error']}")
            return
        
        # System info
        system_info = status.get('system_info', {})
        print(f"📊 System Running: {'✅' if system_info.get('is_running') else '❌'}")
        print(f"⏱️  Uptime: {system_info.get('uptime_seconds', 0):.0f}s")
        print(f"🤖 Agents: {system_info.get('agents_initialized', 0)}")
        print(f"🧠 AMAPI: {'✅' if system_info.get('amapi_integrated') else '❌'}")
        
        # Orchestration metrics
        metrics = status.get('orchestration_metrics', {})
        print(f"\n📈 Orchestration Metrics:")
        print(f"  • Total Workflows: {metrics.get('total_workflows_executed', 0)}")
        print(f"  • Successful: {metrics.get('successful_workflows', 0)}")
        print(f"  • Failed: {metrics.get('failed_workflows', 0)}")
        print(f"  • Avg Duration: {metrics.get('average_execution_time', 0):.2f}s")
        
        # Workflow status
        workflow_status = status.get('workflow_status', {})
        print(f"\n🔄 Workflow Status:")
        print(f"  • Active: {workflow_status.get('active_workflows', 0)}")
        print(f"  • Completed: {workflow_status.get('completed_workflows', 0)}")
        print(f"  • Queued: {workflow_status.get('queued_workflows', 0)}")

    def print_performance_report(self, report: Dict[str, Any]) -> None:
        """Print formatted performance report"""
        print("\n" + "=" * 60)
        print("📊 Performance Report")
        print("=" * 60)
        
        if 'error' in report:
            print(f"❌ Error: {report['error']}")
            return
        
        # Workflow analysis
        workflow_analysis = report.get('workflow_analysis', {})
        print(f"🧪 Workflow Analysis:")
        print(f"  • Success Rate: {workflow_analysis.get('success_rate', 0):.1%}")
        print(f"  • Average Duration: {workflow_analysis.get('average_duration', 0):.2f}s")
        print(f"  • Total Executed: {workflow_analysis.get('total_executed', 0)}")
        
        # System health
        health_score = report.get('system_health_score', 0)
        print(f"❤️  System Health: {health_score:.1%}")
        
        # AMAPI insights
        amapi_insights = report.get('amapi_insights', {})
        if amapi_insights and 'system_metrics' in amapi_insights:
            metrics = amapi_insights['system_metrics']
            print(f"\n🧠 AMAPI Metrics:")
            print(f"  • System IQ: {metrics.get('system_intelligence_quotient', 0):.3f}")
            print(f"  • Collaboration: {metrics.get('collaborative_efficiency_index', 0):.3f}")
            print(f"  • Resilience: {metrics.get('adaptive_resilience_score', 0):.3f}")
            print(f"  • Precision: {metrics.get('predictive_precision_rating', 0):.3f}")

    def print_amapi_analytics(self, analytics: Dict[str, Any]) -> None:
        """Print formatted AMAPI analytics"""
        print("\n" + "=" * 60)
        print("🧠 AMAPI Learning Analytics")
        print("=" * 60)
        
        if 'error' in analytics:
            print(f"❌ Error: {analytics['error']}")
            return
        
        # Learning events
        learning_summary = analytics.get('learning_events_summary', {})
        print(f"📚 Learning Events:")
        print(f"  • Total Events: {learning_summary.get('total_events', 0)}")
        print(f"  • Recent Rate: {learning_summary.get('recent_event_rate', 0):.2f}/min")
        
        events_by_type = learning_summary.get('events_by_type', {})
        for event_type, count in events_by_type.items():
            print(f"  • {event_type}: {count}")
        
        # Knowledge graph
        knowledge_graph = analytics.get('knowledge_graph_analysis', {})
        print(f"\n🕸️  Knowledge Graph:")
        print(f"  • Total Connections: {knowledge_graph.get('total_connections', 0)}")
        print(f"  • Flow Efficiency: {knowledge_graph.get('knowledge_flow_efficiency', 0):.3f}")
        
        # Behavioral models
        behavioral_summary = analytics.get('behavioral_model_summary', {})
        print(f"\n🎭 Behavioral Models:")
        print(f"  • Agents Modeled: {behavioral_summary.get('agents_modeled', 0)}")
        print(f"  • Avg Specialization: {behavioral_summary.get('average_specialization', 0):.3f}")

    async def shutdown_system(self) -> None:
        """Shutdown the QA system"""
        if self.orchestrator:
            logger.info("🛑 Shutting down system...")
            await self.orchestrator.shutdown_system()
            self.orchestrator = None
            logger.info("✅ System shutdown completed")
        else:
            logger.warning("System not running")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent QA System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s wifi                    # Run WiFi toggle test
  %(prog)s settings               # Run settings navigation test
  %(prog)s app Camera             # Run Camera app launch test
  %(prog)s suite                  # Run comprehensive QA suite
  %(prog)s custom "Test volume controls"  # Run custom task
  %(prog)s benchmark --iterations 5      # Run performance benchmark
  %(prog)s status                 # Show system status
  %(prog)s performance            # Show performance report
  %(prog)s amapi                  # Show AMAPI analytics
        """
    )
    
    parser.add_argument(
        'command',
        choices=['wifi', 'settings', 'app', 'suite', 'custom', 'benchmark', 'status', 'performance', 'amapi'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Command arguments (e.g., app name for "app" command)'
    )
    
    parser.add_argument(
        '--device-type',
        choices=['emulator', 'physical', 'simulated'],
        default='emulator',
        help='Device type for testing (default: emulator)'
    )
    
    parser.add_argument(
        '--device-id',
        help='Specific device ID to use'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations for benchmark (default: 10)'
    )
    
    parser.add_argument(
        '--priority',
        type=int,
        choices=range(1, 11),
        default=5,
        help='Task priority for custom tasks (1-10, default: 5)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Expected duration for custom tasks (default: 60.0s)'
    )
    
    parser.add_argument(
        '--output',
        choices=['console', 'json'],
        default='console',
        help='Output format (default: console)'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = QASystemCLI()
    
    try:
        # Initialize system for commands that need it
        if args.command not in ['status', 'performance', 'amapi']:
            await cli.initialize_system(args.device_type, args.device_id)
        
        # Execute command
        if args.command == 'wifi':
            result = await cli.run_wifi_test()
            
        elif args.command == 'settings':
            result = await cli.run_settings_test()
            
        elif args.command == 'app':
            if not args.args:
                print("Error: App name required for 'app' command")
                sys.exit(1)
            app_name = args.args[0]
            result = await cli.run_app_test(app_name)
            
        elif args.command == 'suite':
            result = await cli.run_comprehensive_suite()
            
        elif args.command == 'custom':
            if not args.args:
                print("Error: Task description required for 'custom' command")
                sys.exit(1)
            task_description = ' '.join(args.args)
            result = await cli.run_custom_task(task_description, args.priority, args.duration)
            
        elif args.command == 'benchmark':
            result = await cli.run_benchmark(args.iterations)
            
        elif args.command == 'status':
            result = await cli.get_system_status()
            if args.output == 'console':
                cli.print_system_status(result)
            
        elif args.command == 'performance':
            result = await cli.get_performance_report()
            if args.output == 'console':
                cli.print_performance_report(result)
            
        elif args.command == 'amapi':
            result = await cli.get_amapi_analytics()
            if args.output == 'console':
                cli.print_amapi_analytics(result)
        
        # Output JSON if requested
        if args.output == 'json':
            print(json.dumps(result, indent=2, default=str))
        
        # Shutdown system
        await cli.shutdown_system()
        
    except KeyboardInterrupt:
        logger.info("⚠️  Interrupted by user")
        await cli.shutdown_system()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ CLI execution failed: {e}")
        await cli.shutdown_system()
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run CLI
    asyncio.run(main())