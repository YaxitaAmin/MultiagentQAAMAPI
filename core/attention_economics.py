"""
Attention Economics Engine - Core AMAPI Component
Manages attention allocation, optimization, and economics across agents
"""

import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from loguru import logger

from core.logger import AMAPILogger, LogCategory


class AttentionPoolType(Enum):
    """Types of attention pools"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY = "memory"
    EXECUTION = "execution"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    MONITORING = "monitoring"


@dataclass
class AttentionPool:
    """Individual attention pool"""
    pool_type: AttentionPoolType
    capacity: float
    current_usage: float
    efficiency: float
    recharge_rate: float
    last_updated: float


@dataclass
class AttentionAllocation:
    """Attention allocation result"""
    allocation_id: str
    agent_id: str
    total_attention: float
    pool_allocations: Dict[str, float]
    efficiency_score: float
    timestamp: float
    expected_duration: float = 0.0
    actual_duration: float = 0.0
    success_rate: float = 0.0


@dataclass
class AttentionTransaction:
    """Attention usage transaction"""
    transaction_id: str
    agent_id: str
    allocation_id: str
    attention_used: float
    task_type: str
    success: bool
    efficiency: float
    timestamp: float
    duration: float


class AttentionEconomicsEngine:
    """
    Advanced Attention Economics Engine
    Manages attention as a limited resource with economic principles
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Agent attention pools
        self.agent_pools: Dict[str, Dict[str, AttentionPool]] = {}
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Attention transactions
        self.transactions: List[AttentionTransaction] = []
        self.active_allocations: Dict[str, AttentionAllocation] = {}
        
        # Economic parameters
        self.base_attention_capacity = self.config.get('base_capacity', 10.0)
        self.recharge_rate = self.config.get('recharge_rate', 1.0)  # per second
        self.efficiency_decay = self.config.get('efficiency_decay', 0.1)
        self.collaboration_bonus = self.config.get('collaboration_bonus', 0.2)
        
        # Performance tracking
        self.engine_metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'total_attention_used': 0.0,
            'average_efficiency': 0.0,
            'attention_waste': 0.0,
            'recharge_events': 0
        }
        
        # Logger
        self.logger = AMAPILogger("AttentionEconomics")
        
        # Background tasks
        self._recharge_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        self.logger.info("Attention Economics Engine initialized")

    async def start_engine(self):
        """Start the attention economics engine"""
        try:
            # Start background tasks
            self._recharge_task = asyncio.create_task(self._continuous_recharge())
            self._optimization_task = asyncio.create_task(self._continuous_optimization())
            
            self.logger.info("Attention Economics Engine started")
            
        except Exception as e:
            self.logger.error(f"Error starting attention engine: {e}")
            raise

    async def stop_engine(self):
        """Stop the attention economics engine"""
        try:
            # Cancel background tasks
            if self._recharge_task:
                self._recharge_task.cancel()
            if self._optimization_task:
                self._optimization_task.cancel()
            
            self.logger.info("Attention Economics Engine stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping attention engine: {e}")

    async def initialize_agent_attention(self, agent_id: str, 
                                       pool_configurations: Dict[str, float] = None):
        """Initialize attention pools for an agent"""
        try:
            if pool_configurations is None:
                pool_configurations = {
                    'perception': 2.0,
                    'reasoning': 3.0,
                    'memory': 1.5,
                    'execution': 2.5,
                    'learning': 1.0,
                    'collaboration': 1.0,
                    'monitoring': 0.5
                }
            
            # Create attention pools
            agent_pools = {}
            for pool_name, capacity in pool_configurations.items():
                pool_type = AttentionPoolType(pool_name)
                
                agent_pools[pool_name] = AttentionPool(
                    pool_type=pool_type,
                    capacity=capacity,
                    current_usage=0.0,
                    efficiency=1.0,
                    recharge_rate=self.recharge_rate,
                    last_updated=time.time()
                )
            
            self.agent_pools[agent_id] = agent_pools
            
            # Initialize agent metrics
            self.agent_metrics[agent_id] = {
                'total_attention_used': 0.0,
                'allocations_made': 0,
                'successful_tasks': 0,
                'average_efficiency': 1.0,
                'collaboration_events': 0,
                'learning_events': 0,
                'last_activity': time.time()
            }
            
            self.logger.info(f"Initialized attention pools for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error initializing agent attention: {e}")
            raise

    async def allocate_attention_for_task(self, allocation_request: Dict[str, Any]) -> AttentionAllocation:
        """Allocate attention for a specific task"""
        try:
            agent_id = allocation_request['agent_id']
            task_description = allocation_request['task_description']
            task_complexity = allocation_request.get('task_complexity', 0.5)
            
            if agent_id not in self.agent_pools:
                await self.initialize_agent_attention(agent_id)
            
            # Calculate attention requirements
            attention_requirements = self._calculate_attention_requirements(
                task_description, task_complexity, allocation_request
            )
            
            # Check availability and allocate
            allocation = await self._perform_allocation(
                agent_id, attention_requirements, allocation_request
            )
            
            # Store allocation
            self.active_allocations[allocation.allocation_id] = allocation
            
            # Update metrics
            self.engine_metrics['total_allocations'] += 1
            self.agent_metrics[agent_id]['allocations_made'] += 1
            
            # Log allocation
            self.logger.log_attention_event(
                agent_id, 
                {
                    'event_type': 'attention_allocated',
                    'allocation_id': allocation.allocation_id,
                    'total_attention': allocation.total_attention,
                    'efficiency_score': allocation.efficiency_score
                }
            )
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error allocating attention: {e}")
            # Return minimal allocation
            return AttentionAllocation(
                allocation_id=f"fallback_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                total_attention=1.0,
                pool_allocations={'reasoning': 1.0},
                efficiency_score=0.5,
                timestamp=time.time()
            )

    def _calculate_attention_requirements(self, task_description: str, 
                                        task_complexity: float,
                                        allocation_request: Dict[str, Any]) -> Dict[str, float]:
        """Calculate attention requirements for a task"""
        try:
            # Base requirements by task type
            task_type = self._classify_task_type(task_description)
            
            base_requirements = {
                'planning': {
                    'reasoning': 3.0, 'memory': 1.5, 'perception': 1.0,
                    'execution': 0.5, 'learning': 0.5, 'collaboration': 1.0
                },
                'execution': {
                    'execution': 3.0, 'perception': 2.0, 'reasoning': 1.5,
                    'memory': 1.0, 'monitoring': 1.0, 'learning': 0.5
                },
                'verification': {
                    'monitoring': 2.5, 'reasoning': 2.0, 'perception': 2.0,
                    'memory': 1.5, 'execution': 0.5, 'learning': 0.5
                },
                'supervision': {
                    'monitoring': 3.0, 'collaboration': 2.5, 'reasoning': 2.0,
                    'perception': 1.5, 'memory': 1.0, 'learning': 1.0
                },
                'learning': {
                    'learning': 3.0, 'memory': 2.5, 'reasoning': 2.0,
                    'perception': 1.0, 'execution': 0.5, 'collaboration': 1.0
                }
            }
            
            # Get base requirements
            requirements = base_requirements.get(task_type, {
                'reasoning': 2.0, 'perception': 1.5, 'execution': 1.5,
                'memory': 1.0, 'learning': 0.5, 'collaboration': 0.5
            })
            
            # Scale by complexity
            complexity_multiplier = 0.5 + (task_complexity * 1.0)
            scaled_requirements = {
                pool: req * complexity_multiplier 
                for pool, req in requirements.items()
            }
            
            # Add specialization bonuses
            agent_type = allocation_request.get('agent_type', 'general')
            specializations = allocation_request.get('specializations', [])
            
            for specialization in specializations:
                if specialization in task_description.lower():
                    # Reduce attention needed for specialized tasks
                    for pool in scaled_requirements:
                        scaled_requirements[pool] *= 0.9
            
            return scaled_requirements
            
        except Exception as e:
            self.logger.error(f"Error calculating attention requirements: {e}")
            return {'reasoning': 2.0, 'execution': 1.0}

    def _classify_task_type(self, task_description: str) -> str:
        """Classify task type from description"""
        description_lower = task_description.lower()
        
        if any(word in description_lower for word in ['plan', 'strategy', 'analyze', 'design']):
            return 'planning'
        elif any(word in description_lower for word in ['execute', 'perform', 'run', 'action']):
            return 'execution'
        elif any(word in description_lower for word in ['verify', 'check', 'validate', 'test']):
            return 'verification'
        elif any(word in description_lower for word in ['supervise', 'coordinate', 'manage', 'oversee']):
            return 'supervision'
        elif any(word in description_lower for word in ['learn', 'adapt', 'improve', 'train']):
            return 'learning'
        else:
            return 'general'

    async def _perform_allocation(self, agent_id: str, requirements: Dict[str, float],
                                allocation_request: Dict[str, Any]) -> AttentionAllocation:
        """Perform the actual attention allocation"""
        try:
            agent_pools = self.agent_pools[agent_id]
            allocation_id = f"alloc_{uuid.uuid4().hex[:8]}"
            
            # Check availability and calculate actual allocation
            pool_allocations = {}
            total_attention = 0.0
            efficiency_penalties = []
            
            for pool_name, required_attention in requirements.items():
                if pool_name in agent_pools:
                    pool = agent_pools[pool_name]
                    
                    # Calculate available capacity
                    available = pool.capacity - pool.current_usage
                    
                    # Determine actual allocation
                    actual_allocation = min(required_attention, available)
                    
                    if actual_allocation < required_attention:
                        # Calculate efficiency penalty for insufficient attention
                        shortage_ratio = (required_attention - actual_allocation) / required_attention
                        efficiency_penalties.append(shortage_ratio)
                    
                    pool_allocations[pool_name] = actual_allocation
                    total_attention += actual_allocation
                    
                    # Update pool usage
                    pool.current_usage += actual_allocation
                    pool.last_updated = time.time()
            
            # Calculate overall efficiency score
            if efficiency_penalties:
                efficiency_score = max(0.1, 1.0 - np.mean(efficiency_penalties))
            else:
                efficiency_score = 1.0
            
            # Apply collaboration bonus if applicable
            if allocation_request.get('collaboration_context'):
                efficiency_score = min(1.0, efficiency_score + self.collaboration_bonus)
            
            allocation = AttentionAllocation(
                allocation_id=allocation_id,
                agent_id=agent_id,
                total_attention=total_attention,
                pool_allocations=pool_allocations,
                efficiency_score=efficiency_score,
                timestamp=time.time()
            )
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error performing allocation: {e}")
            raise

    async def update_attention_usage(self, agent_id: str, attention_used: float, 
                                   success: bool, task_type: str = "general",
                                   allocation_id: str = None):
        """Update attention usage after task completion"""
        try:
            # Create transaction record
            transaction = AttentionTransaction(
                transaction_id=f"trans_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                allocation_id=allocation_id or "unknown",
                attention_used=attention_used,
                task_type=task_type,
                success=success,
                efficiency=1.0 if success else 0.5,  # Simplified efficiency calculation
                timestamp=time.time(),
                duration=0.0  # Would be calculated from allocation timestamp
            )
            
            self.transactions.append(transaction)
            
            # Update agent metrics
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]
                metrics['total_attention_used'] += attention_used
                metrics['last_activity'] = time.time()
                
                if success:
                    metrics['successful_tasks'] += 1
                
                # Update average efficiency
                total_tasks = metrics['allocations_made']
                if total_tasks > 0:
                    current_avg = metrics['average_efficiency']
                    new_efficiency = transaction.efficiency
                    metrics['average_efficiency'] = (
                        (current_avg * (total_tasks - 1) + new_efficiency) / total_tasks
                    )
            
            # Update engine metrics
            self.engine_metrics['total_attention_used'] += attention_used
            if success:
                self.engine_metrics['successful_allocations'] += 1
            
            # Calculate attention waste
            if allocation_id and allocation_id in self.active_allocations:
                allocation = self.active_allocations[allocation_id]
                allocated_attention = allocation.total_attention
                
                if attention_used < allocated_attention:
                    waste = allocated_attention - attention_used
                    self.engine_metrics['attention_waste'] += waste
                
                # Update allocation with actual usage
                allocation.actual_duration = time.time() - allocation.timestamp
                allocation.success_rate = 1.0 if success else 0.0
            
            # Log transaction
            self.logger.log_attention_event(
                agent_id,
                {
                    'event_type': 'attention_used',
                    'transaction_id': transaction.transaction_id,
                    'attention_used': attention_used,
                    'success': success,
                    'efficiency': transaction.efficiency
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error updating attention usage: {e}")

    async def _continuous_recharge(self):
        """Continuously recharge attention pools"""
        try:
            while True:
                await asyncio.sleep(1.0)  # Recharge every second
                
                current_time = time.time()
                
                for agent_id, pools in self.agent_pools.items():
                    for pool_name, pool in pools.items():
                        # Calculate recharge amount
                        time_delta = current_time - pool.last_updated
                        recharge_amount = pool.recharge_rate * time_delta
                        
                        # Apply recharge
                        new_usage = max(0.0, pool.current_usage - recharge_amount)
                        
                        if new_usage != pool.current_usage:
                            pool.current_usage = new_usage
                            pool.last_updated = current_time
                            
                            if pool.current_usage == 0.0:
                                self.engine_metrics['recharge_events'] += 1
                
        except asyncio.CancelledError:
            self.logger.info("Attention recharge task cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous recharge: {e}")

    async def _continuous_optimization(self):
        """Continuously optimize attention allocation strategies"""
        try:
            while True:
                await asyncio.sleep(30.0)  # Optimize every 30 seconds
                
                # Analyze recent performance
                await self._analyze_attention_patterns()
                
                # Optimize pool configurations
                await self._optimize_pool_configurations()
                
                # Clean up old data
                await self._cleanup_old_data()
                
        except asyncio.CancelledError:
            self.logger.info("Attention optimization task cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous optimization: {e}")

    async def _analyze_attention_patterns(self):
        """Analyze attention usage patterns for optimization"""
        try:
            # Analyze recent transactions
            recent_transactions = [
                t for t in self.transactions 
                if time.time() - t.timestamp < 3600  # Last hour
            ]
            
            if not recent_transactions:
                return
            
            # Calculate efficiency by task type
            task_type_efficiency = {}
            for transaction in recent_transactions:
                task_type = transaction.task_type
                if task_type not in task_type_efficiency:
                    task_type_efficiency[task_type] = []
                task_type_efficiency[task_type].append(transaction.efficiency)
            
            # Update engine metrics
            all_efficiencies = [t.efficiency for t in recent_transactions]
            if all_efficiencies:
                self.engine_metrics['average_efficiency'] = np.mean(all_efficiencies)
            
            # Log analysis results
            self.logger.debug(f"Attention pattern analysis: {len(recent_transactions)} transactions analyzed")
            
        except Exception as e:
            self.logger.error(f"Error analyzing attention patterns: {e}")

    async def _optimize_pool_configurations(self):
        """Optimize attention pool configurations based on usage patterns"""
        try:
            # Analyze pool usage patterns
            for agent_id, pools in self.agent_pools.items():
                # Calculate utilization rates
                utilization_rates = {}
                for pool_name, pool in pools.items():
                    utilization_rate = pool.current_usage / pool.capacity
                    utilization_rates[pool_name] = utilization_rate
                
                # Identify over/under-utilized pools
                over_utilized = {k: v for k, v in utilization_rates.items() if v > 0.8}
                under_utilized = {k: v for k, v in utilization_rates.items() if v < 0.2}
                
                # Apply optimizations (simplified)
                for pool_name in over_utilized:
                    pool = pools[pool_name]
                    pool.capacity = min(pool.capacity * 1.1, self.base_attention_capacity * 2)
                
                if over_utilized or under_utilized:
                    self.logger.debug(f"Optimized attention pools for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing pool configurations: {e}")

    async def _cleanup_old_data(self):
        """Clean up old transaction and allocation data"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600 * 24  # 24 hours
            
            # Clean up old transactions
            old_count = len(self.transactions)
            self.transactions = [
                t for t in self.transactions 
                if t.timestamp > cutoff_time
            ]
            
            # Clean up old allocations
            old_allocations = []
            for alloc_id, allocation in self.active_allocations.items():
                if allocation.timestamp < cutoff_time:
                    old_allocations.append(alloc_id)
            
            for alloc_id in old_allocations:
                del self.active_allocations[alloc_id]
            
            if len(self.transactions) < old_count or old_allocations:
                self.logger.debug(f"Cleaned up {old_count - len(self.transactions)} transactions and {len(old_allocations)} allocations")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    async def cleanup_agent_attention(self, agent_id: str):
        """Clean up attention data for an agent"""
        try:
            if agent_id in self.agent_pools:
                del self.agent_pools[agent_id]
            
            if agent_id in self.agent_metrics:
                del self.agent_metrics[agent_id]
            
            # Remove agent transactions
            self.transactions = [
                t for t in self.transactions 
                if t.agent_id != agent_id
            ]
            
            # Remove agent allocations
            agent_allocations = [
                alloc_id for alloc_id, allocation in self.active_allocations.items()
                if allocation.agent_id == agent_id
            ]
            
            for alloc_id in agent_allocations:
                del self.active_allocations[alloc_id]
            
            self.logger.info(f"Cleaned up attention data for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up agent attention: {e}")

    async def get_current_attention_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current attention state for an agent"""
        try:
            if agent_id not in self.agent_pools:
                return {'error': 'Agent not found'}
            
            pools = self.agent_pools[agent_id]
            metrics = self.agent_metrics.get(agent_id, {})
            
            pool_states = {}
            for pool_name, pool in pools.items():
                pool_states[pool_name] = {
                    'capacity': pool.capacity,
                    'current_usage': pool.current_usage,
                    'available': pool.capacity - pool.current_usage,
                    'utilization_rate': pool.current_usage / pool.capacity,
                    'efficiency': pool.efficiency,
                    'recharge_rate': pool.recharge_rate
                }
            
            return {
                'agent_id': agent_id,
                'pools': pool_states,
                'metrics': metrics,
                'active_allocations': len([
                    a for a in self.active_allocations.values() 
                    if a.agent_id == agent_id
                ]),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting attention state: {e}")
            return {'error': str(e)}

    def calculate_attention_cost(self, task_parameters: Dict[str, Any]) -> float:
        """Calculate attention cost for a task"""
        try:
            task_type = task_parameters.get('action_type', 'general')
            complexity = task_parameters.get('complexity', 0.5)
            
            # Base costs by task type
            base_costs = {
                'agent_reasoning': 2.5,
                'ui_interaction': 1.5,
                'verification': 2.0,
                'planning': 3.0,
                'learning': 2.5,
                'collaboration': 1.0
            }
            
            base_cost = base_costs.get(task_type, 2.0)
            
            # Scale by complexity
            complexity_multiplier = 0.5 + (complexity * 1.0)
            
            attention_cost = base_cost * complexity_multiplier
            
            return attention_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating attention cost: {e}")
            return 2.0  # Default cost

    def get_engine_analytics(self) -> Dict[str, Any]:
        """Get comprehensive engine analytics"""
        try:
            # Calculate recent activity
            hour_ago = time.time() - 3600
            recent_transactions = [
                t for t in self.transactions 
                if t.timestamp > hour_ago
            ]
            
            recent_allocations = [
                a for a in self.active_allocations.values()
                if a.timestamp > hour_ago
            ]
            
            # Agent statistics
            agent_stats = {}
            for agent_id, metrics in self.agent_metrics.items():
                agent_stats[agent_id] = {
                    'total_attention_used': metrics['total_attention_used'],
                    'average_efficiency': metrics['average_efficiency'],
                    'successful_tasks': metrics['successful_tasks'],
                    'allocations_made': metrics['allocations_made']
                }
            
            return {
                'engine_metrics': self.engine_metrics.copy(),
                'active_agents': len(self.agent_pools),
                'active_allocations': len(self.active_allocations),
                'total_transactions': len(self.transactions),
                'recent_activity': {
                    'transactions_last_hour': len(recent_transactions),
                    'allocations_last_hour': len(recent_allocations)
                },
                'agent_statistics': agent_stats,
                'system_health': {
                    'average_pool_utilization': self._calculate_average_utilization(),
                    'attention_waste_rate': self._calculate_waste_rate(),
                    'efficiency_trend': self._calculate_efficiency_trend()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating engine analytics: {e}")
            return {'error': str(e)}

    def _calculate_average_utilization(self) -> float:
        """Calculate average pool utilization across all agents"""
        try:
            utilizations = []
            for pools in self.agent_pools.values():
                for pool in pools.values():
                    if pool.capacity > 0:
                        utilizations.append(pool.current_usage / pool.capacity)
            
            return np.mean(utilizations) if utilizations else 0.0
            
        except Exception as e:
            return 0.0

    def _calculate_waste_rate(self) -> float:
        """Calculate attention waste rate"""
        try:
            total_allocated = sum(
                a.total_attention for a in self.active_allocations.values()
            )
            
            if total_allocated > 0:
                return self.engine_metrics['attention_waste'] / total_allocated
            else:
                return 0.0
                
        except Exception as e:
            return 0.0

    def _calculate_efficiency_trend(self) -> str:
        """Calculate efficiency trend"""
        try:
            recent_transactions = [
                t for t in self.transactions[-50:]  # Last 50 transactions
                if t.timestamp > time.time() - 3600
            ]
            
            if len(recent_transactions) < 10:
                return 'insufficient_data'
            
            # Split into first and second half
            mid_point = len(recent_transactions) // 2
            first_half = recent_transactions[:mid_point]
            second_half = recent_transactions[mid_point:]
            
            first_avg = np.mean([t.efficiency for t in first_half])
            second_avg = np.mean([t.efficiency for t in second_half])
            
            if second_avg > first_avg + 0.05:
                return 'improving'
            elif second_avg < first_avg - 0.05:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            return 'unknown'


__all__ = [
    "AttentionEconomicsEngine",
    "AttentionAllocation",
    "AttentionPool",
    "AttentionTransaction",
    "AttentionPoolType"
]