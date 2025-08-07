"""
Behavioral Learning Engine - Advanced pattern recognition and learning
Learns from agent interactions and optimizes behavioral patterns
"""

import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict
from loguru import logger

from core.logger import AMAPILogger, LogCategory


class LearningType(Enum):
    """Types of learning events"""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    OPTIMIZATION = "optimization"
    ADAPTATION = "adaptation"
    COLLABORATION = "collaboration"
    ATTENTION_EFFICIENCY = "attention_efficiency"


class PatternType(Enum):
    """Types of behavioral patterns"""
    ACTION_SEQUENCE = "action_sequence"
    DECISION_TREE = "decision_tree"
    ATTENTION_ALLOCATION = "attention_allocation"
    ERROR_RECOVERY = "error_recovery"
    COLLABORATION_PATTERN = "collaboration_pattern"
    TIMING_PATTERN = "timing_pattern"


@dataclass
class BehavioralPattern:
    """Behavioral pattern structure"""
    pattern_id: str
    pattern_type: PatternType
    pattern_data: Dict[str, Any]
    success_rate: float
    usage_count: int
    confidence: float
    context_conditions: List[str]
    learned_from: List[str]  # Agent IDs
    created_timestamp: float
    last_updated: float
    effectiveness_score: float


@dataclass
class LearningEvent:
    """Learning event structure"""
    event_id: str
    learning_type: LearningType
    agent_id: str
    event_data: Dict[str, Any]
    outcome: str
    performance_metrics: Dict[str, float]
    timestamp: float
    pattern_updates: List[str]  # Pattern IDs updated


@dataclass
class LearningInsight:
    """Learning insight derived from patterns"""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    supporting_patterns: List[str]
    recommended_actions: List[str]
    impact_score: float
    generated_timestamp: float


class BehavioralPatternEngine:
    """
    Advanced Behavioral Pattern Learning Engine
    Learns from agent behaviors and optimizes performance
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Pattern storage
        self.patterns: Dict[str, BehavioralPattern] = {}
        self.learning_events: List[LearningEvent] = []
        self.insights: List[LearningInsight] = []
        
        # Learning parameters
        self.min_pattern_confidence = self.config.get('min_confidence', 0.7)
        self.pattern_decay_rate = self.config.get('decay_rate', 0.05)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        
        # Agent-specific learning data
        self.agent_learning_profiles: Dict[str, Dict[str, Any]] = {}
        self.collaboration_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.learning_metrics = {
            'patterns_learned': 0,
            'patterns_applied': 0,
            'successful_applications': 0,
            'learning_events_processed': 0,
            'insights_generated': 0,
            'average_pattern_confidence': 0.0,
            'learning_efficiency': 0.0
        }
        
        # Logger
        self.logger = AMAPILogger("BehavioralLearning")
        
        # Background learning task
        self._learning_task: Optional[asyncio.Task] = None
        
        self.logger.info("Behavioral Pattern Engine initialized")

    async def start_learning_engine(self):
        """Start the behavioral learning engine"""
        try:
            self._learning_task = asyncio.create_task(self._continuous_learning())
            self.logger.info("Behavioral learning engine started")
        except Exception as e:
            self.logger.error(f"Error starting learning engine: {e}")
            raise

    async def stop_learning_engine(self):
        """Stop the behavioral learning engine"""
        try:
            if self._learning_task:
                self._learning_task.cancel()
            self.logger.info("Behavioral learning engine stopped")
        except Exception as e:
            self.logger.error(f"Error stopping learning engine: {e}")

    async def record_learning_event(self, agent_id: str, learning_type: LearningType,
                                  event_data: Dict[str, Any], outcome: str,
                                  performance_metrics: Dict[str, float] = None) -> str:
        """Record a learning event for pattern extraction"""
        try:
            event_id = f"learn_{uuid.uuid4().hex[:8]}"
            
            learning_event = LearningEvent(
                event_id=event_id,
                learning_type=learning_type,
                agent_id=agent_id,
                event_data=event_data,
                outcome=outcome,
                performance_metrics=performance_metrics or {},
                timestamp=time.time(),
                pattern_updates=[]
            )
            
            self.learning_events.append(learning_event)
            self.learning_metrics['learning_events_processed'] += 1
            
            # Initialize agent learning profile if needed
            if agent_id not in self.agent_learning_profiles:
                self.agent_learning_profiles[agent_id] = {
                    'learning_events': 0,
                    'successful_patterns': 0,
                    'specializations': [],
                    'learning_velocity': 0.0,
                    'adaptation_score': 0.5
                }
            
            self.agent_learning_profiles[agent_id]['learning_events'] += 1
            
            # Process learning event for pattern extraction
            await self._process_learning_event(learning_event)
            
            # Log learning event
            self.logger.log_learning_event(
                agent_id,
                learning_type.value,
                event_data,
                performance_metrics.get('improvement', 0.0) if performance_metrics else 0.0
            )
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Error recording learning event: {e}")
            return ""

    async def _process_learning_event(self, learning_event: LearningEvent):
        """Process learning event to extract or update patterns"""
        try:
            # Extract potential patterns based on learning type
            if learning_event.learning_type == LearningType.SUCCESS_PATTERN:
                await self._extract_success_pattern(learning_event)
            
            elif learning_event.learning_type == LearningType.FAILURE_PATTERN:
                await self._extract_failure_pattern(learning_event)
            
            elif learning_event.learning_type == LearningType.ATTENTION_EFFICIENCY:
                await self._extract_attention_pattern(learning_event)
            
            elif learning_event.learning_type == LearningType.COLLABORATION:
                await self._extract_collaboration_pattern(learning_event)
            
            # Update existing patterns with new data
            await self._update_related_patterns(learning_event)
            
        except Exception as e:
            self.logger.error(f"Error processing learning event: {e}")

    async def _extract_success_pattern(self, learning_event: LearningEvent):
        """Extract success patterns from learning event"""
        try:
            event_data = learning_event.event_data
            
            # Look for action sequences that led to success
            if 'action_sequence' in event_data and 'success_rate' in event_data:
                success_rate = event_data['success_rate']
                
                if success_rate > 0.8:  # High success threshold
                    pattern_data = {
                        'action_sequence': event_data['action_sequence'],
                        'context': event_data.get('context', {}),
                        'conditions': event_data.get('conditions', []),
                        'timing': event_data.get('timing', {}),
                        'resources_used': event_data.get('resources_used', {})
                    }
                    
                    # Create or update pattern
                    pattern = await self._create_or_update_pattern(
                        PatternType.ACTION_SEQUENCE,
                        pattern_data,
                        success_rate,
                        [learning_event.agent_id]
                    )
                    
                    learning_event.pattern_updates.append(pattern.pattern_id)
                    self.logger.debug(f"Extracted success pattern: {pattern.pattern_id}")
            
        except Exception as e:
            self.logger.error(f"Error extracting success pattern: {e}")

    async def _extract_failure_pattern(self, learning_event: LearningEvent):
        """Extract failure patterns for error recovery"""
        try:
            event_data = learning_event.event_data
            
            # Look for failure conditions and recovery strategies
            if 'failure_type' in event_data and 'recovery_actions' in event_data:
                pattern_data = {
                    'failure_type': event_data['failure_type'],
                    'failure_conditions': event_data.get('conditions', []),
                    'recovery_actions': event_data['recovery_actions'],
                    'recovery_success_rate': event_data.get('recovery_success_rate', 0.5),
                    'context': event_data.get('context', {})
                }
                
                # Create error recovery pattern
                pattern = await self._create_or_update_pattern(
                    PatternType.ERROR_RECOVERY,
                    pattern_data,
                    event_data.get('recovery_success_rate', 0.5),
                    [learning_event.agent_id]
                )
                
                learning_event.pattern_updates.append(pattern.pattern_id)
                self.logger.debug(f"Extracted failure recovery pattern: {pattern.pattern_id}")
            
        except Exception as e:
            self.logger.error(f"Error extracting failure pattern: {e}")

    async def _extract_attention_pattern(self, learning_event: LearningEvent):
        """Extract attention allocation patterns"""
        try:
            event_data = learning_event.event_data
            
            if 'attention_allocation' in event_data and 'efficiency_score' in event_data:
                efficiency = event_data['efficiency_score']
                
                if efficiency > 0.8:  # High efficiency threshold
                    pattern_data = {
                        'attention_allocation': event_data['attention_allocation'],
                        'task_type': event_data.get('task_type', 'general'),
                        'complexity': event_data.get('complexity', 0.5),
                        'efficiency_score': efficiency,
                        'context': event_data.get('context', {})
                    }
                    
                    pattern = await self._create_or_update_pattern(
                        PatternType.ATTENTION_ALLOCATION,
                        pattern_data,
                        efficiency,
                        [learning_event.agent_id]
                    )
                    
                    learning_event.pattern_updates.append(pattern.pattern_id)
                    self.logger.debug(f"Extracted attention pattern: {pattern.pattern_id}")
            
        except Exception as e:
            self.logger.error(f"Error extracting attention pattern: {e}")

    async def _extract_collaboration_pattern(self, learning_event: LearningEvent):
        """Extract collaboration patterns"""
        try:
            event_data = learning_event.event_data
            
            if 'collaboration_type' in event_data and 'participants' in event_data:
                collaboration_success = event_data.get('success_rate', 0.5)
                
                pattern_data = {
                    'collaboration_type': event_data['collaboration_type'],
                    'participants': event_data['participants'],
                    'roles': event_data.get('roles', {}),
                    'communication_pattern': event_data.get('communication_pattern', []),
                    'success_factors': event_data.get('success_factors', []),
                    'duration': event_data.get('duration', 0.0)
                }
                
                pattern = await self._create_or_update_pattern(
                    PatternType.COLLABORATION_PATTERN,
                    pattern_data,
                    collaboration_success,
                    event_data['participants']
                )
                
                learning_event.pattern_updates.append(pattern.pattern_id)
                
                # Store collaboration pattern for cross-agent learning
                collab_key = '_'.join(sorted(event_data['participants']))
                if collab_key not in self.collaboration_patterns:
                    self.collaboration_patterns[collab_key] = []
                
                self.collaboration_patterns[collab_key].append({
                    'pattern_id': pattern.pattern_id,
                    'success_rate': collaboration_success,
                    'timestamp': time.time()
                })
                
                self.logger.debug(f"Extracted collaboration pattern: {pattern.pattern_id}")
            
        except Exception as e:
            self.logger.error(f"Error extracting collaboration pattern: {e}")

    async def _create_or_update_pattern(self, pattern_type: PatternType,
                                      pattern_data: Dict[str, Any],
                                      success_rate: float,
                                      learned_from: List[str]) -> BehavioralPattern:
        """Create new pattern or update existing one"""
        try:
            # Check if similar pattern exists
            similar_pattern = await self._find_similar_pattern(pattern_type, pattern_data)
            
            if similar_pattern:
                # Update existing pattern
                pattern = similar_pattern
                
                # Update success rate with weighted average
                old_weight = pattern.usage_count
                new_weight = 1
                total_weight = old_weight + new_weight
                
                pattern.success_rate = (
                    (pattern.success_rate * old_weight + success_rate * new_weight) / total_weight
                )
                
                pattern.usage_count += 1
                pattern.last_updated = time.time()
                pattern.confidence = min(1.0, pattern.confidence + self.learning_rate)
                
                # Update effectiveness score
                pattern.effectiveness_score = self._calculate_effectiveness_score(pattern)
                
                # Add new agents to learned_from
                for agent_id in learned_from:
                    if agent_id not in pattern.learned_from:
                        pattern.learned_from.append(agent_id)
                
            else:
                # Create new pattern
                pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
                
                pattern = BehavioralPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    pattern_data=pattern_data,
                    success_rate=success_rate,
                    usage_count=1,
                    confidence=0.6,  # Initial confidence
                    context_conditions=pattern_data.get('conditions', []),
                    learned_from=learned_from.copy(),
                    created_timestamp=time.time(),
                    last_updated=time.time(),
                    effectiveness_score=success_rate
                )
                
                self.patterns[pattern_id] = pattern
                self.learning_metrics['patterns_learned'] += 1
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error creating/updating pattern: {e}")
            raise

    async def _find_similar_pattern(self, pattern_type: PatternType,
                                  pattern_data: Dict[str, Any]) -> Optional[BehavioralPattern]:
        """Find similar existing pattern"""
        try:
            for pattern in self.patterns.values():
                if pattern.pattern_type != pattern_type:
                    continue
                
                # Calculate similarity based on pattern type
                similarity = self._calculate_pattern_similarity(pattern.pattern_data, pattern_data)
                
                if similarity > 0.8:  # High similarity threshold
                    return pattern
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding similar pattern: {e}")
            return None

    def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], 
                                    pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns"""
        try:
            # Simple similarity calculation based on common keys and values
            common_keys = set(pattern1.keys()) & set(pattern2.keys())
            if not common_keys:
                return 0.0
            
            matches = 0
            total_comparisons = 0
            
            for key in common_keys:
                if key in ['context', 'conditions']:
                    # Skip complex nested comparisons for now
                    continue
                
                total_comparisons += 1
                
                val1, val2 = pattern1[key], pattern2[key]
                
                if isinstance(val1, (str, int, float)) and isinstance(val2, (str, int, float)):
                    if val1 == val2:
                        matches += 1
                elif isinstance(val1, list) and isinstance(val2, list):
                    # Check list overlap
                    if set(val1) == set(val2):
                        matches += 1
                    elif len(set(val1) & set(val2)) > 0:
                        matches += 0.5
            
            if total_comparisons == 0:
                return 0.0
            
            return matches / total_comparisons
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

    def _calculate_effectiveness_score(self, pattern: BehavioralPattern) -> float:
        """Calculate pattern effectiveness score"""
        try:
            # Combine success rate, usage count, and confidence
            usage_factor = min(1.0, pattern.usage_count / 10.0)  # Normalize to 0-1
            
            effectiveness = (
                pattern.success_rate * 0.5 +
                pattern.confidence * 0.3 +
                usage_factor * 0.2
            )
            
            return effectiveness
            
        except Exception as e:
            self.logger.error(f"Error calculating effectiveness score: {e}")
            return 0.5

    async def _update_related_patterns(self, learning_event: LearningEvent):
        """Update patterns related to the learning event"""
        try:
            # Find patterns that might be affected by this learning event
            related_patterns = []
            
            for pattern in self.patterns.values():
                if learning_event.agent_id in pattern.learned_from:
                    # Check if the learning event context matches pattern conditions
                    if self._matches_pattern_context(learning_event.event_data, pattern):
                        related_patterns.append(pattern)
            
            # Update related patterns
            for pattern in related_patterns:
                # Apply small adjustments based on learning event outcome
                outcome_success = learning_event.outcome == 'success'
                performance_score = learning_event.performance_metrics.get('score', 0.5)
                
                if outcome_success and performance_score > 0.7:
                    pattern.confidence = min(1.0, pattern.confidence + 0.05)
                elif not outcome_success:
                    pattern.confidence = max(0.1, pattern.confidence - 0.02)
                
                pattern.last_updated = time.time()
                pattern.effectiveness_score = self._calculate_effectiveness_score(pattern)
                
                learning_event.pattern_updates.append(pattern.pattern_id)
            
        except Exception as e:
            self.logger.error(f"Error updating related patterns: {e}")

    def _matches_pattern_context(self, event_data: Dict[str, Any], 
                                pattern: BehavioralPattern) -> bool:
        """Check if event data matches pattern context"""
        try:
            # Simple context matching
            event_context = event_data.get('context', {})
            pattern_conditions = pattern.context_conditions
            
            if not pattern_conditions:
                return True
            
            # Check if any conditions match
            for condition in pattern_conditions:
                if condition in str(event_context):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error matching pattern context: {e}")
            return False

    async def get_relevant_patterns(self, agent_id: str, context: Dict[str, Any],
                                  pattern_type: PatternType = None) -> List[BehavioralPattern]:
        """Get relevant patterns for an agent in a specific context"""
        try:
            relevant_patterns = []
            
            for pattern in self.patterns.values():
                # Filter by pattern type if specified
                if pattern_type and pattern.pattern_type != pattern_type:
                    continue
                
                # Check if pattern is relevant to agent
                if agent_id in pattern.learned_from or pattern.confidence > 0.8:
                    # Check context relevance
                    if self._is_pattern_relevant(pattern, context):
                        relevant_patterns.append(pattern)
            
            # Sort by effectiveness score
            relevant_patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)
            
            return relevant_patterns
            
        except Exception as e:
            self.logger.error(f"Error getting relevant patterns: {e}")
            return []

    def _is_pattern_relevant(self, pattern: BehavioralPattern, context: Dict[str, Any]) -> bool:
        """Check if pattern is relevant to current context"""
        try:
            # Check confidence threshold
            if pattern.confidence < self.min_pattern_confidence:
                return False
            
            # Check context conditions
            if pattern.context_conditions:
                context_str = str(context)
                matches = sum(1 for condition in pattern.context_conditions 
                            if condition in context_str)
                relevance_score = matches / len(pattern.context_conditions)
                
                return relevance_score > 0.5
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking pattern relevance: {e}")
            return False

    async def apply_pattern(self, pattern_id: str, agent_id: str, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a learned pattern in a specific context"""
        try:
            if pattern_id not in self.patterns:
                return {'success': False, 'error': 'Pattern not found'}
            
            pattern = self.patterns[pattern_id]
            
            # Check if pattern is applicable
            if not self._is_pattern_relevant(pattern, context):
                return {'success': False, 'error': 'Pattern not relevant to context'}
            
            # Update usage metrics
            pattern.usage_count += 1
            pattern.last_updated = time.time()
            self.learning_metrics['patterns_applied'] += 1
            
            # Return pattern application result
            application_result = {
                'success': True,
                'pattern_id': pattern_id,
                'pattern_type': pattern.pattern_type.value,
                'pattern_data': pattern.pattern_data,
                'success_rate': pattern.success_rate,
                'confidence': pattern.confidence,
                'application_context': context,
                'applied_by': agent_id,
                'timestamp': time.time()
            }
            
            self.logger.debug(f"Applied pattern {pattern_id} by agent {agent_id}")
            
            return application_result
            
        except Exception as e:
            self.logger.error(f"Error applying pattern: {e}")
            return {'success': False, 'error': str(e)}

    async def _continuous_learning(self):
        """Continuous learning and pattern optimization"""
        try:
            while True:
                await asyncio.sleep(60.0)  # Run every minute
                
                # Decay old patterns
                await self._decay_patterns()
                
                # Generate insights
                await self._generate_insights()
                
                # Optimize patterns
                await self._optimize_patterns()
                
                # Update agent learning profiles
                await self._update_agent_profiles()
                
        except asyncio.CancelledError:
            self.logger.info("Continuous learning task cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous learning: {e}")

    async def _decay_patterns(self):
        """Apply decay to old patterns"""
        try:
            current_time = time.time()
            
            for pattern in self.patterns.values():
                # Calculate age in hours
                age_hours = (current_time - pattern.last_updated) / 3600
                
                if age_hours > 24:  # Start decay after 24 hours
                    decay_factor = self.pattern_decay_rate * (age_hours - 24) / 24
                    pattern.confidence = max(0.1, pattern.confidence - decay_factor)
                    pattern.effectiveness_score = self._calculate_effectiveness_score(pattern)
            
        except Exception as e:
            self.logger.error(f"Error in pattern decay: {e}")

    async def _generate_insights(self):
        """Generate learning insights from patterns"""
        try:
            # Generate insights from high-performing patterns
            high_performance_patterns = [
                p for p in self.patterns.values() 
                if p.effectiveness_score > 0.8 and p.usage_count > 5
            ]
            
            for pattern in high_performance_patterns[:5]:  # Top 5 patterns
                insight = self._create_insight_from_pattern(pattern)
                if insight:
                    self.insights.append(insight)
                    self.learning_metrics['insights_generated'] += 1
            
            # Keep only recent insights
            cutoff_time = time.time() - 3600 * 24  # 24 hours
            self.insights = [
                insight for insight in self.insights 
                if insight.generated_timestamp > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")

    def _create_insight_from_pattern(self, pattern: BehavioralPattern) -> Optional[LearningInsight]:
        """Create insight from a high-performing pattern"""
        try:
            insight_id = f"insight_{uuid.uuid4().hex[:8]}"
            
            if pattern.pattern_type == PatternType.ACTION_SEQUENCE:
                description = f"Action sequence with {pattern.success_rate:.2f} success rate is highly effective"
                recommended_actions = ["Apply this sequence in similar contexts", "Share with other agents"]
            
            elif pattern.pattern_type == PatternType.ATTENTION_ALLOCATION:
                description = f"Attention allocation pattern shows {pattern.effectiveness_score:.2f} efficiency"
                recommended_actions = ["Use this allocation for similar tasks", "Train other agents"]
            
            elif pattern.pattern_type == PatternType.COLLABORATION_PATTERN:
                description = f"Collaboration pattern between {len(pattern.learned_from)} agents is successful"
                recommended_actions = ["Encourage similar collaborations", "Document best practices"]
            
            else:
                description = f"Pattern {pattern.pattern_id} shows high effectiveness"
                recommended_actions = ["Continue using this pattern", "Monitor performance"]
            
            return LearningInsight(
                insight_id=insight_id,
                insight_type=pattern.pattern_type.value,
                description=description,
                confidence=pattern.confidence,
                supporting_patterns=[pattern.pattern_id],
                recommended_actions=recommended_actions,
                impact_score=pattern.effectiveness_score,
                generated_timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error creating insight: {e}")
            return None

    async def _optimize_patterns(self):
        """Optimize patterns for better performance"""
        try:
            # Remove low-confidence patterns
            patterns_to_remove = []
            for pattern_id, pattern in self.patterns.items():
                if pattern.confidence < 0.2 and pattern.usage_count < 2:
                    patterns_to_remove.append(pattern_id)
            
            for pattern_id in patterns_to_remove:
                del self.patterns[pattern_id]
            
            if patterns_to_remove:
                self.logger.debug(f"Removed {len(patterns_to_remove)} low-confidence patterns")
            
            # Update average pattern confidence
            if self.patterns:
                avg_confidence = sum(p.confidence for p in self.patterns.values()) / len(self.patterns)
                self.learning_metrics['average_pattern_confidence'] = avg_confidence
            
        except Exception as e:
            self.logger.error(f"Error optimizing patterns: {e}")

    async def _update_agent_profiles(self):
        """Update agent learning profiles"""
        try:
            for agent_id, profile in self.agent_learning_profiles.items():
                # Calculate learning velocity
                recent_events = [
                    event for event in self.learning_events
                    if event.agent_id == agent_id and 
                    time.time() - event.timestamp < 3600  # Last hour
                ]
                
                profile['learning_velocity'] = len(recent_events)
                
                # Update adaptation score
                agent_patterns = [
                    p for p in self.patterns.values() 
                    if agent_id in p.learned_from
                ]
                
                if agent_patterns:
                    avg_effectiveness = sum(p.effectiveness_score for p in agent_patterns) / len(agent_patterns)
                    profile['adaptation_score'] = avg_effectiveness
                    profile['successful_patterns'] = sum(
                        1 for p in agent_patterns if p.effectiveness_score > 0.7
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating agent profiles: {e}")

    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive learning analytics"""
        try:
            # Pattern statistics
            pattern_stats = {
                'total_patterns': len(self.patterns),
                'patterns_by_type': {},
                'high_confidence_patterns': 0,
                'average_effectiveness': 0.0
            }
            
            for pattern in self.patterns.values():
                pattern_type = pattern.pattern_type.value
                pattern_stats['patterns_by_type'][pattern_type] = pattern_stats['patterns_by_type'].get(pattern_type, 0) + 1
                
                if pattern.confidence > 0.8:
                    pattern_stats['high_confidence_patterns'] += 1
            
            if self.patterns:
                pattern_stats['average_effectiveness'] = sum(p.effectiveness_score for p in self.patterns.values()) / len(self.patterns)
            
            # Learning efficiency
            if self.learning_metrics['patterns_applied'] > 0:
                self.learning_metrics['learning_efficiency'] = (
                    self.learning_metrics['successful_applications'] / 
                    self.learning_metrics['patterns_applied']
                )
            
            return {
                'learning_metrics': self.learning_metrics.copy(),
                'pattern_statistics': pattern_stats,
                'agent_profiles': {
                    agent_id: profile.copy() 
                    for agent_id, profile in self.agent_learning_profiles.items()
                },
                'insights_generated': len(self.insights),
                'collaboration_patterns': len(self.collaboration_patterns),
                'total_learning_events': len(self.learning_events)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating learning analytics: {e}")
            return {'error': str(e)}

    def get_agent_learning_status(self, agent_id: str) -> Dict[str, Any]:
        """Get learning status for specific agent"""
        try:
            if agent_id not in self.agent_learning_profiles:
                return {'error': 'Agent not found'}
            
            profile = self.agent_learning_profiles[agent_id]
            
            # Get agent's patterns
            agent_patterns = [
                {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type.value,
                    'success_rate': p.success_rate,
                    'confidence': p.confidence,
                    'usage_count': p.usage_count
                }
                for p in self.patterns.values() 
                if agent_id in p.learned_from
            ]
            
            return {
                'agent_id': agent_id,
                'learning_profile': profile.copy(),
                'patterns_learned': len(agent_patterns),
                'patterns': agent_patterns,
                'recent_insights': [
                    asdict(insight) for insight in self.insights
                    if any(agent_id in self.patterns[pid].learned_from 
                          for pid in insight.supporting_patterns 
                          if pid in self.patterns)
                ][-5:]  # Last 5 insights
            }
            
        except Exception as e:
            self.logger.error(f"Error getting agent learning status: {e}")
            return {'error': str(e)}


__all__ = [
    "BehavioralPatternEngine",
    "BehavioralPattern",
    "LearningEvent",
    "LearningInsight",
    "LearningType",
    "PatternType"
]