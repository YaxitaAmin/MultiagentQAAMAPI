"""
AMAPI Logger - Advanced Multi-Agent Performance Intelligence Logging System
Provides comprehensive logging with structured data and performance tracking
"""

import time
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger
import sys


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log category enumeration"""
    AGENT_ACTION = "agent_action"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE = "performance"
    ATTENTION = "attention"
    COLLABORATION = "collaboration"
    LEARNING = "learning"
    ERROR = "error"
    METRICS = "metrics"


@dataclass
class LogEntry:
    """Structured log entry"""
    log_id: str
    timestamp: float
    level: LogLevel
    category: LogCategory
    source: str
    message: str
    data: Dict[str, Any]
    execution_context: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class AMAPILogger:
    """
    Advanced Multi-Agent Performance Intelligence Logger
    Provides structured logging with performance tracking and analytics
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        
        # Log storage
        self.log_entries: List[LogEntry] = []
        self.log_buffer: List[LogEntry] = []
        self.max_entries = self.config.get('max_entries', 10000)
        self.buffer_size = self.config.get('buffer_size', 100)
        
        # Performance tracking
        self.performance_metrics = {
            'total_logs': 0,
            'logs_by_level': {level.value: 0 for level in LogLevel},
            'logs_by_category': {category.value: 0 for category in LogCategory},
            'average_log_size': 0.0,
            'logging_overhead': 0.0
        }
        
        # Setup loguru logger
        self._setup_loguru()
        
        # Initialize file handlers if configured
        self._setup_file_logging()
        
        self.info(f"AMAPI Logger initialized: {name}")

    def _setup_loguru(self):
        """Setup loguru logger with AMAPI configuration"""
        try:
            # Remove default handler
            logger.remove()
            
            # Add console handler with custom format
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            
            logger.add(
                sys.stdout,
                format=log_format,
                level=self.config.get('console_level', 'INFO'),
                colorize=True,
                enqueue=True
            )
            
        except Exception as e:
            print(f"Error setting up loguru: {e}")

    def _setup_file_logging(self):
        """Setup file logging if configured"""
        try:
            log_dir = self.config.get('log_directory', 'logs')
            if log_dir:
                log_path = Path(log_dir)
                log_path.mkdir(exist_ok=True)
                
                # Main log file
                main_log_file = log_path / f"{self.name}.log"
                logger.add(
                    str(main_log_file),
                    format="{time} | {level} | {name} | {message}",
                    level=self.config.get('file_level', 'DEBUG'),
                    rotation="10 MB",
                    retention="7 days",
                    compression="zip",
                    enqueue=True
                )
                
                # Error log file
                error_log_file = log_path / f"{self.name}_errors.log"
                logger.add(
                    str(error_log_file),
                    format="{time} | {level} | {name} | {message}",
                    level="ERROR",
                    rotation="5 MB",
                    retention="14 days",
                    compression="zip",
                    enqueue=True
                )
                
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")

    def _create_log_entry(self, level: LogLevel, category: LogCategory, 
                         message: str, data: Dict[str, Any] = None,
                         execution_context: Dict[str, Any] = None,
                         performance_metrics: Dict[str, Any] = None) -> LogEntry:
        """Create structured log entry"""
        return LogEntry(
            log_id=f"log_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            level=level,
            category=category,
            source=self.name,
            message=message,
            data=data or {},
            execution_context=execution_context,
            performance_metrics=performance_metrics
        )

    def _log_entry(self, entry: LogEntry):
        """Process and store log entry"""
        start_time = time.time()
        
        try:
            # Add to buffer
            self.log_buffer.append(entry)
            
            # Update performance metrics
            self.performance_metrics['total_logs'] += 1
            self.performance_metrics['logs_by_level'][entry.level.value] += 1
            self.performance_metrics['logs_by_category'][entry.category.value] += 1
            
            # Calculate average log size
            entry_size = len(json.dumps(asdict(entry), default=str))
            total_logs = self.performance_metrics['total_logs']
            current_avg = self.performance_metrics['average_log_size']
            self.performance_metrics['average_log_size'] = (
                (current_avg * (total_logs - 1) + entry_size) / total_logs
            )
            
            # Flush buffer if needed
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_buffer()
            
            # Log to loguru
            loguru_logger = logger.bind(category=entry.category.value)
            
            if entry.level == LogLevel.DEBUG:
                loguru_logger.debug(entry.message)
            elif entry.level == LogLevel.INFO:
                loguru_logger.info(entry.message)
            elif entry.level == LogLevel.WARNING:
                loguru_logger.warning(entry.message)
            elif entry.level == LogLevel.ERROR:
                loguru_logger.error(entry.message)
            elif entry.level == LogLevel.CRITICAL:
                loguru_logger.critical(entry.message)
            
            # Update logging overhead
            overhead = time.time() - start_time
            current_overhead = self.performance_metrics['logging_overhead']
            self.performance_metrics['logging_overhead'] = (
                (current_overhead * (total_logs - 1) + overhead) / total_logs
            )
            
        except Exception as e:
            # Fallback logging
            print(f"Logging error: {e}")

    def _flush_buffer(self):
        """Flush log buffer to permanent storage"""
        try:
            # Move buffer entries to main storage
            self.log_entries.extend(self.log_buffer)
            self.log_buffer.clear()
            
            # Trim old entries if needed
            if len(self.log_entries) > self.max_entries:
                excess = len(self.log_entries) - self.max_entries
                self.log_entries = self.log_entries[excess:]
            
        except Exception as e:
            print(f"Error flushing log buffer: {e}")

    # Public logging methods
    def debug(self, message: str, data: Dict[str, Any] = None, 
             category: LogCategory = LogCategory.SYSTEM_EVENT,
             execution_context: Dict[str, Any] = None):
        """Log debug message"""
        entry = self._create_log_entry(
            LogLevel.DEBUG, category, message, data, execution_context
        )
        self._log_entry(entry)

    def info(self, message: str, data: Dict[str, Any] = None,
            category: LogCategory = LogCategory.SYSTEM_EVENT,
            execution_context: Dict[str, Any] = None):
        """Log info message"""
        entry = self._create_log_entry(
            LogLevel.INFO, category, message, data, execution_context
        )
        self._log_entry(entry)

    def warning(self, message: str, data: Dict[str, Any] = None,
               category: LogCategory = LogCategory.SYSTEM_EVENT,
               execution_context: Dict[str, Any] = None):
        """Log warning message"""
        entry = self._create_log_entry(
            LogLevel.WARNING, category, message, data, execution_context
        )
        self._log_entry(entry)

    def error(self, message: str, data: Dict[str, Any] = None,
             category: LogCategory = LogCategory.ERROR,
             execution_context: Dict[str, Any] = None):
        """Log error message"""
        entry = self._create_log_entry(
            LogLevel.ERROR, category, message, data, execution_context
        )
        self._log_entry(entry)

    def critical(self, message: str, data: Dict[str, Any] = None,
                category: LogCategory = LogCategory.ERROR,
                execution_context: Dict[str, Any] = None):
        """Log critical message"""
        entry = self._create_log_entry(
            LogLevel.CRITICAL, category, message, data, execution_context
        )
        self._log_entry(entry)

    # Specialized logging methods
    def log_agent_action(self, agent_id: str, action_type: str, 
                        action_data: Dict[str, Any], success: bool,
                        execution_time: float, attention_cost: float = 0.0):
        """Log agent action with performance metrics"""
        performance_metrics = {
            'execution_time': execution_time,
            'attention_cost': attention_cost,
            'success': success
        }
        
        data = {
            'agent_id': agent_id,
            'action_type': action_type,
            'action_data': action_data,
            'success': success
        }
        
        level = LogLevel.INFO if success else LogLevel.WARNING
        message = f"Agent {agent_id} executed {action_type}: {'SUCCESS' if success else 'FAILED'}"
        
        entry = self._create_log_entry(
            level, LogCategory.AGENT_ACTION, message, data,
            performance_metrics=performance_metrics
        )
        self._log_entry(entry)

    def log_performance(self, component: str, metrics: Dict[str, Any]):
        """Log performance metrics"""
        message = f"Performance metrics for {component}"
        entry = self._create_log_entry(
            LogLevel.INFO, LogCategory.PERFORMANCE, message, metrics
        )
        self._log_entry(entry)

    def log_attention_event(self, agent_id: str, attention_data: Dict[str, Any]):
        """Log attention economics event"""
        message = f"Attention event for agent {agent_id}"
        data = {'agent_id': agent_id, **attention_data}
        entry = self._create_log_entry(
            LogLevel.DEBUG, LogCategory.ATTENTION, message, data
        )
        self._log_entry(entry)

    def log_collaboration(self, agents: List[str], collaboration_type: str,
                         collaboration_data: Dict[str, Any], success: bool):
        """Log collaboration event"""
        message = f"Collaboration {collaboration_type} between {', '.join(agents)}: {'SUCCESS' if success else 'FAILED'}"
        data = {
            'agents': agents,
            'collaboration_type': collaboration_type,
            'collaboration_data': collaboration_data,
            'success': success
        }
        
        level = LogLevel.INFO if success else LogLevel.WARNING
        entry = self._create_log_entry(
            level, LogCategory.COLLABORATION, message, data
        )
        self._log_entry(entry)

    def log_learning_event(self, agent_id: str, learning_type: str,
                          learning_data: Dict[str, Any], improvement: float):
        """Log learning event"""
        message = f"Learning event for agent {agent_id}: {learning_type} (improvement: {improvement:.3f})"
        data = {
            'agent_id': agent_id,
            'learning_type': learning_type,
            'learning_data': learning_data,
            'improvement': improvement
        }
        
        entry = self._create_log_entry(
            LogLevel.INFO, LogCategory.LEARNING, message, data
        )
        self._log_entry(entry)

    # Analytics methods
    def get_log_analytics(self) -> Dict[str, Any]:
        """Get comprehensive log analytics"""
        try:
            # Ensure buffer is flushed
            if self.log_buffer:
                self._flush_buffer()
            
            total_entries = len(self.log_entries)
            if total_entries == 0:
                return {'no_data': True}
            
            # Time-based analytics
            timestamps = [entry.timestamp for entry in self.log_entries]
            time_span = max(timestamps) - min(timestamps)
            
            # Category distribution
            category_distribution = {}
            for category in LogCategory:
                count = sum(1 for entry in self.log_entries if entry.category == category)
                category_distribution[category.value] = count
            
            # Level distribution
            level_distribution = {}
            for level in LogLevel:
                count = sum(1 for entry in self.log_entries if entry.level == level)
                level_distribution[level.value] = count
            
            # Recent activity (last hour)
            hour_ago = time.time() - 3600
            recent_entries = [entry for entry in self.log_entries if entry.timestamp > hour_ago]
            
            # Error rate
            error_entries = [entry for entry in self.log_entries 
                           if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
            error_rate = len(error_entries) / total_entries if total_entries > 0 else 0
            
            return {
                'total_entries': total_entries,
                'time_span_hours': time_span / 3600,
                'category_distribution': category_distribution,
                'level_distribution': level_distribution,
                'recent_activity_count': len(recent_entries),
                'error_rate': error_rate,
                'performance_metrics': self.performance_metrics.copy(),
                'logger_name': self.name
            }
            
        except Exception as e:
            logger.error(f"Error generating log analytics: {e}")
            return {'error': str(e)}

    def query_logs(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[LogEntry]:
        """Query logs with filters"""
        try:
            # Ensure buffer is flushed
            if self.log_buffer:
                self._flush_buffer()
            
            results = self.log_entries.copy()
            
            if filters:
                # Apply filters
                if 'level' in filters:
                    level_filter = LogLevel(filters['level'])
                    results = [entry for entry in results if entry.level == level_filter]
                
                if 'category' in filters:
                    category_filter = LogCategory(filters['category'])
                    results = [entry for entry in results if entry.category == category_filter]
                
                if 'source' in filters:
                    source_filter = filters['source']
                    results = [entry for entry in results if source_filter in entry.source]
                
                if 'start_time' in filters:
                    start_time = filters['start_time']
                    results = [entry for entry in results if entry.timestamp >= start_time]
                
                if 'end_time' in filters:
                    end_time = filters['end_time']
                    results = [entry for entry in results if entry.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error querying logs: {e}")
            return []

    def export_logs(self, filepath: str, format: str = 'json') -> bool:
        """Export logs to file"""
        try:
            # Ensure buffer is flushed
            if self.log_buffer:
                self._flush_buffer()
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    log_data = {
                        'export_timestamp': time.time(),
                        'logger_name': self.name,
                        'performance_metrics': self.performance_metrics,
                        'entries': [asdict(entry) for entry in self.log_entries]
                    }
                    json.dump(log_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import csv
                with open(filepath, 'w', newline='') as f:
                    fieldnames = ['timestamp', 'level', 'category', 'source', 'message']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for entry in self.log_entries:
                        writer.writerow({
                            'timestamp': entry.timestamp,
                            'level': entry.level.value,
                            'category': entry.category.value,
                            'source': entry.source,
                            'message': entry.message
                        })
            
            self.info(f"Logs exported to {filepath} in {format} format")
            return True
            
        except Exception as e:
            self.error(f"Error exporting logs: {e}")
            return False

    def clear_logs(self, keep_recent_hours: int = 24):
        """Clear old logs, keeping recent entries"""
        try:
            cutoff_time = time.time() - (keep_recent_hours * 3600)
            
            # Keep recent entries
            self.log_entries = [
                entry for entry in self.log_entries 
                if entry.timestamp > cutoff_time
            ]
            
            # Clear buffer
            self.log_buffer.clear()
            
            self.info(f"Cleared logs older than {keep_recent_hours} hours")
            
        except Exception as e:
            self.error(f"Error clearing logs: {e}")

    def get_logger_status(self) -> Dict[str, Any]:
        """Get logger status and health"""
        return {
            'logger_name': self.name,
            'total_entries': len(self.log_entries),
            'buffer_size': len(self.log_buffer),
            'performance_metrics': self.performance_metrics.copy(),
            'config': self.config.copy(),
            'is_healthy': len(self.log_entries) < self.max_entries
        }


# Global logger instance
_global_logger = None

def get_global_logger() -> AMAPILogger:
    """Get global AMAPI logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AMAPILogger("AMAPI_Global")
    return _global_logger


__all__ = [
    "AMAPILogger",
    "LogLevel",
    "LogCategory", 
    "LogEntry",
    "get_global_logger"
]