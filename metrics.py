"""
System Metrics Collection and Analysis
Comprehensive metrics tracking for AMAPI system performance
"""

import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from pathlib import Path
from loguru import logger

from core.logger import AMAPILogger, LogCategory


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = None


@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    unit: str
    points: deque
    aggregations: Dict[str, float]
    last_updated: float


class SystemMetrics:
    """
    Comprehensive system metrics collection and analysis
    Tracks performance, health, and operational metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = AMAPILogger("SystemMetrics")
        
        # Metrics storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.metric_history_limit = self.config.get('history_limit', 1000)
        
        # Collection intervals
        self.collection_interval = self.config.get('collection_interval', 5.0)  # seconds
        self.aggregation_interval = self.config.get('aggregation_interval', 60.0)  # seconds
        
        # Export settings
        self.export_enabled = self.config.get('export_enabled', True)
        self.export_directory = Path(self.config.get('export_directory', 'metrics'))
        self.export_directory.mkdir(exist_ok=True)
        
        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        
        # System start time
        self.system_start_time = time.time()
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        self.logger.info("System Metrics initialized")

    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        core_metrics = [
            ('system_uptime', 'seconds'),
            ('cpu_usage', 'percent'),
            ('memory_usage', 'bytes'),
            ('task_completion_rate', 'rate'),
            ('task_success_rate', 'percent'),
            ('average_task_duration', 'seconds'),
            ('agent_response_time', 'seconds'),
            ('attention_efficiency', 'score'),
            ('behavioral_learning_rate', 'rate'),
            ('device_compatibility_score', 'score'),
            ('llm_response_time', 'seconds'),
            ('error_rate', 'percent'),
            ('system_health_score', 'score'),
            ('coordination_efficiency', 'score'),
            ('resource_utilization', 'percent')
        ]
        
        for metric_name, unit in core_metrics:
            self.metrics[metric_name] = MetricSeries(
                name=metric_name,
                unit=unit,
                points=deque(maxlen=self.metric_history_limit),
                aggregations={},
                last_updated=time.time()
            )

    async def initialize(self):
        """Initialize metrics collection"""
        try:
            # Start background collection
            self._collection_task = asyncio.create_task(self._continuous_collection())
            self._aggregation_task = asyncio.create_task(self._continuous_aggregation())
            
            self.logger.info("Metrics collection started")
            
        except Exception as e:
            self.logger.error(f"Error initializing metrics: {e}")
            raise

    async def record_metric(self, name: str, value: float, 
                           metadata: Dict[str, Any] = None, unit: str = None):
        """Record a metric value"""
        try:
            # Create metric series if it doesn't exist
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    unit=unit or 'unknown',
                    points=deque(maxlen=self.metric_history_limit),
                    aggregations={},
                    last_updated=time.time()
                )
            
            # Add metric point
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                metadata=metadata or {}
            )
            
            self.metrics[name].points.append(metric_point)
            self.metrics[name].last_updated = time.time()
            
            # Update real-time aggregations
            await self._update_metric_aggregations(name)
            
        except Exception as e:
            self.logger.error(f"Error recording metric {name}: {e}")

    async def _update_metric_aggregations(self, metric_name: str):
        """Update aggregations for a metric"""
        try:
            metric_series = self.metrics[metric_name]
            values = [point.value for point in metric_series.points]
            
            if not values:
                return
            
            # Calculate aggregations
            aggregations = {
                'count': len(values),
                'latest': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'sum': sum(values)
            }
            
            # Add additional aggregations for sufficient data
            if len(values) > 1:
                aggregations['std_dev'] = statistics.stdev(values)
                aggregations['median'] = statistics.median(values)
            
            if len(values) >= 4:
                # Percentiles
                sorted_values = sorted(values)
                aggregations['p95'] = sorted_values[int(0.95 * len(sorted_values))]
                aggregations['p99'] = sorted_values[int(0.99 * len(sorted_values))]
            
            metric_series.aggregations = aggregations
            
        except Exception as e:
            self.logger.error(f"Error updating aggregations for {metric_name}: {e}")

    async def get_metric(self, name: str, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Get metric data with optional time range filtering"""
        try:
            if name not in self.metrics:
                return {'error': f'Metric {name} not found'}
            
            metric_series = self.metrics[name]
            points = list(metric_series.points)
            
            # Apply time range filter if specified
            if time_range:
                start_time, end_time = time_range
                points = [
                    point for point in points
                    if start_time <= point.timestamp <= end_time
                ]
            
            return {
                'name': name,
                'unit': metric_series.unit,
                'points_count': len(points),
                'aggregations': metric_series.aggregations.copy(),
                'last_updated': metric_series.last_updated,
                'points': [asdict(point) for point in points] if len(points) <= 100 else 'truncated'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metric {name}: {e}")
            return {'error': str(e)}

    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics summary"""
        try:
            metrics_summary = {}
            
            for name, metric_series in self.metrics.items():
                metrics_summary[name] = {
                    'unit': metric_series.unit,
                    'points_count': len(metric_series.points),
                    'aggregations': metric_series.aggregations.copy(),
                    'last_updated': metric_series.last_updated
                }
            
            return {
                'total_metrics': len(self.metrics),
                'collection_uptime': time.time() - self.system_start_time,
                'metrics': metrics_summary,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all metrics: {e}")
            return {'error': str(e)}

    async def _continuous_collection(self):
        """Continuous metrics collection"""
        try:
            while True:
                await asyncio.sleep(self.collection_interval)
                
                # Collect system metrics
                await self._collect_system_metrics()
                
        except asyncio.CancelledError:
            self.logger.info("Metrics collection task cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous collection: {e}")

    async def _collect_system_metrics(self):
        """Collect core system metrics"""
        try:
            current_time = time.time()
            
            # System uptime
            await self.record_metric('system_uptime', current_time - self.system_start_time)
            
            # Basic system health score (placeholder)
            await self.record_metric('system_health_score', 0.95)
            
            # Resource utilization (placeholder)
            await self.record_metric('resource_utilization', 65.0)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def _continuous_aggregation(self):
        """Continuous metrics aggregation and export"""
        try:
            while True:
                await asyncio.sleep(self.aggregation_interval)
                
                # Update all aggregations
                for metric_name in self.metrics:
                    await self._update_metric_aggregations(metric_name)
                
                # Export metrics if enabled
                if self.export_enabled:
                    await self._export_metrics()
                
        except asyncio.CancelledError:
            self.logger.info("Metrics aggregation task cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous aggregation: {e}")

    async def _export_metrics(self):
        """Export metrics to files"""
        try:
            timestamp = int(time.time())
            export_file = self.export_directory / f"metrics_{timestamp}.json"
            
            metrics_data = await self.get_all_metrics()
            
            with open(export_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            # Keep only recent exports
            await self._cleanup_old_exports()
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")

    async def _cleanup_old_exports(self):
        """Clean up old metric export files"""
        try:
            max_exports = self.config.get('max_export_files', 100)
            
            export_files = sorted(
                self.export_directory.glob('metrics_*.json'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Remove old files
            for old_file in export_files[max_exports:]:
                old_file.unlink()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up exports: {e}")

    def record_task_metrics(self, task_duration: float, success: bool, 
                           agent_response_times: Dict[str, float] = None):
        """Record task-specific metrics"""
        try:
            # Task duration
            asyncio.create_task(self.record_metric('task_duration', task_duration))
            
            # Task success
            asyncio.create_task(self.record_metric('task_success', 1.0 if success else 0.0))
            
            # Agent response times
            if agent_response_times:
                for agent, response_time in agent_response_times.items():
                    asyncio.create_task(
                        self.record_metric(f'agent_response_time_{agent}', response_time)
                    )
            
        except Exception as e:
            self.logger.error(f"Error recording task metrics: {e}")

    def record_attention_metrics(self, allocation_efficiency: float, 
                               total_attention_used: float, waste_rate: float):
        """Record attention economics metrics"""
        try:
            asyncio.create_task(self.record_metric('attention_efficiency', allocation_efficiency))
            asyncio.create_task(self.record_metric('attention_used', total_attention_used))
            asyncio.create_task(self.record_metric('attention_waste_rate', waste_rate))
            
        except Exception as e:
            self.logger.error(f"Error recording attention metrics: {e}")

    def record_learning_metrics(self, patterns_learned: int, learning_rate: float,
                              adaptation_success_rate: float):
        """Record behavioral learning metrics"""
        try:
            asyncio.create_task(self.record_metric('patterns_learned', patterns_learned))
            asyncio.create_task(self.record_metric('learning_rate', learning_rate))
            asyncio.create_task(self.record_metric('adaptation_success_rate', adaptation_success_rate))
            
        except Exception as e:
            self.logger.error(f"Error recording learning metrics: {e}")

    def record_device_metrics(self, compatibility_score: float, adaptations_made: int,
                            adaptation_success_rate: float):
        """Record device abstraction metrics"""
        try:
            asyncio.create_task(self.record_metric('device_compatibility', compatibility_score))
            asyncio.create_task(self.record_metric('adaptations_made', adaptations_made))
            asyncio.create_task(self.record_metric('device_adaptation_success_rate', adaptation_success_rate))
            
        except Exception as e:
            self.logger.error(f"Error recording device metrics: {e}")

    def record_llm_metrics(self, response_time: float, tokens_used: int, 
                          confidence: float, provider: str):
        """Record LLM interface metrics"""
        try:
            asyncio.create_task(self.record_metric('llm_response_time', response_time))
            asyncio.create_task(self.record_metric('llm_tokens_used', tokens_used))
            asyncio.create_task(self.record_metric('llm_confidence', confidence))
            asyncio.create_task(
                self.record_metric(f'llm_usage_{provider}', 1.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error recording LLM metrics: {e}")

    async def get_performance_report(self, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'report_generated': time.time(),
                'time_range': time_range,
                'summary': {},
                'performance_indicators': {},
                'trends': {},
                'alerts': []
            }
            
            # Key performance indicators
            key_metrics = [
                'task_success_rate',
                'average_task_duration',
                'attention_efficiency',
                'device_compatibility',
                'system_health_score'
            ]
            
            for metric_name in key_metrics:
                if metric_name in self.metrics:
                    metric_data = await self.get_metric(metric_name, time_range)
                    if 'aggregations' in metric_data:
                        report['performance_indicators'][metric_name] = metric_data['aggregations']
            
            # System summary
            all_metrics = await self.get_all_metrics()
            report['summary'] = {
                'total_metrics_tracked': all_metrics['total_metrics'],
                'collection_uptime': all_metrics['collection_uptime'],
                'data_points_collected': sum(
                    m['points_count'] for m in all_metrics['metrics'].values()
                )
            }
            
            # Generate alerts for concerning metrics
            report['alerts'] = await self._generate_metric_alerts()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}

    async def _generate_metric_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts for concerning metric values"""
        alerts = []
        
        try:
            # Define alert thresholds
            alert_thresholds = {
                'task_success_rate': {'min': 70.0, 'severity': 'high'},
                'attention_efficiency': {'min': 60.0, 'severity': 'medium'},
                'system_health_score': {'min': 80.0, 'severity': 'high'},
                'error_rate': {'max': 10.0, 'severity': 'medium'},
                'average_task_duration': {'max': 120.0, 'severity': 'low'}
            }
            
            for metric_name, thresholds in alert_thresholds.items():
                if metric_name in self.metrics:
                    aggregations = self.metrics[metric_name].aggregations
                    
                    if 'mean' in aggregations:
                        value = aggregations['mean']
                        
                        # Check minimum threshold
                        if 'min' in thresholds and value < thresholds['min']:
                            alerts.append({
                                'metric': metric_name,
                                'type': 'below_threshold',
                                'value': value,
                                'threshold': thresholds['min'],
                                'severity': thresholds['severity'],
                                'timestamp': time.time()
                            })
                        
                        # Check maximum threshold
                        if 'max' in thresholds and value > thresholds['max']:
                            alerts.append({
                                'metric': metric_name,
                                'type': 'above_threshold',
                                'value': value,
                                'threshold': thresholds['max'],
                                'severity': thresholds['severity'],
                                'timestamp': time.time()
                            })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error generating metric alerts: {e}")
            return []

    async def cleanup(self):
        """Cleanup metrics collection"""
        try:
            # Cancel background tasks
            if self._collection_task:
                self._collection_task.cancel()
            if self._aggregation_task:
                self._aggregation_task.cancel()
            
            # Final export
            if self.export_enabled:
                await self._export_metrics()
            
            self.logger.info("System metrics cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during metrics cleanup: {e}")


__all__ = [
    "SystemMetrics",
    "MetricPoint",
    "MetricSeries"
]