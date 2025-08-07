"""
Core AMAPI Learning System
Adaptive Multi-Agent Performance Intelligence for QA System
"""

import time
import json
import uuid
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import numpy as np
from loguru import logger


class LearningType(Enum):
    """Types of learning events"""
    PATTERN_DISCOVERY = "pattern_discovery"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BEHAVIORAL_ADAPTATION = "behavioral_adaptation"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    PREDICTIVE_LEARNING = "predictive_learning"


@dataclass
class AMAPILearningEvent:
    """AMAPI learning event with comprehensive metadata"""
    event_id: str
    event_type: LearningType
    timestamp: float
    agent_id: str
    learning_data: Dict[str, Any]
    impact_score: float
    confidence: float
    validation_status: str  # pending, validated, rejected
    related_events: List[str]
    context: Dict[str, Any]


class AMAPICore:
    """
    Core AMAPI Learning System
    Manages cross-agent learning, pattern recognition, and performance optimization
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Learning storage
        self.learning_events: List[AMAPILearningEvent] = []
        self.global_patterns: Dict[str, Dict[str, Any]] = {}
        self.agent_behavioral_models: Dict[str, Dict[str, Any]] = {}
        self.performance_predictions: Dict[str, Dict[str, Any]] = {}
        
        # Cross-agent learning
        self.knowledge_graph: Dict[str, List[str]] = defaultdict(list)
        self.collaboration_patterns: Dict[str, Dict[str, Any]] = {}
        self.attention_economics: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.system_metrics = {
            'total_learning_events': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'knowledge_transfers': 0,
            'pattern_discoveries': 0,
            'optimization_applications': 0,
            'system_intelligence_quotient': 0.0,
            'collaborative_efficiency_index': 0.0,
            'adaptive_resilience_score': 0.0,
            'predictive_precision_rating': 0.0
        }
        
        logger.info("AMAPI Core Learning System initialized")

    async def register_learning_event(self, agent_id: str, event_type: LearningType,
                                     learning_data: Dict[str, Any],
                                     context: Dict[str, Any] = None) -> str:
        """Register a new learning event"""
        event_id = f"amapi_{uuid.uuid4().hex[:8]}"
        
        event = AMAPILearningEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            agent_id=agent_id,
            learning_data=learning_data,
            impact_score=self._calculate_impact_score(learning_data),
            confidence=learning_data.get('confidence', 0.7),
            validation_status='pending',
            related_events=[],
            context=context or {}
        )
        
        self.learning_events.append(event)
        self.system_metrics['total_learning_events'] += 1
        
        # Process learning event
        await self._process_learning_event(event)
        
        logger.info(f"AMAPI learning event registered: {event_type.value} from {agent_id}")
        return event_id

    async def _process_learning_event(self, event: AMAPILearningEvent) -> None:
        """Process and integrate learning event"""
        try:
            # Update agent behavioral model
            await self._update_agent_behavioral_model(event)
            
            # Discover patterns
            patterns = await self._discover_patterns(event)
            if patterns:
                self.system_metrics['pattern_discoveries'] += len(patterns)
            
            # Update global knowledge
            await self._update_global_knowledge(event, patterns)
            
            # Generate predictions
            predictions = await self._generate_predictions(event)
            if predictions:
                self.performance_predictions[event.agent_id] = predictions
            
            # Facilitate knowledge transfer
            transfers = await self._facilitate_knowledge_transfer(event)
            self.system_metrics['knowledge_transfers'] += len(transfers)
            
            # Update system metrics
            await self._update_system_metrics()
            
        except Exception as e:
            logger.error(f"Error processing learning event: {e}")

    async def _update_agent_behavioral_model(self, event: AMAPILearningEvent) -> None:
        """Update behavioral model for the agent"""
        agent_id = event.agent_id
        
        if agent_id not in self.agent_behavioral_models:
            self.agent_behavioral_models[agent_id] = {
                'learning_patterns': {},
                'success_indicators': {},
                'failure_patterns': {},
                'attention_usage': [],
                'collaboration_preferences': {},
                'adaptation_rate': 0.0,
                'specialization_areas': []
            }
        
        model = self.agent_behavioral_models[agent_id]
        
        # Update based on event type
        if event.event_type == LearningType.PATTERN_DISCOVERY:
            pattern_data = event.learning_data.get('pattern', {})
            pattern_key = pattern_data.get('key', 'unknown')
            model['learning_patterns'][pattern_key] = {
                'frequency': model['learning_patterns'].get(pattern_key, {}).get('frequency', 0) + 1,
                'success_rate': pattern_data.get('success_rate', 0.0),
                'last_seen': event.timestamp
            }
        
        elif event.event_type == LearningType.PERFORMANCE_OPTIMIZATION:
            optimization = event.learning_data.get('optimization', {})
            if optimization.get('success', False):
                model['success_indicators'][optimization.get('type', 'unknown')] = optimization
        
        elif event.event_type == LearningType.BEHAVIORAL_ADAPTATION:
            adaptation = event.learning_data.get('adaptation', {})
            model['adaptation_rate'] = (model['adaptation_rate'] + adaptation.get('rate', 0.0)) / 2
        
        # Update attention usage
        if 'attention_cost' in event.learning_data:
            model['attention_usage'].append({
                'timestamp': event.timestamp,
                'cost': event.learning_data['attention_cost'],
                'efficiency': event.learning_data.get('efficiency', 0.5)
            })
            
            # Keep only recent attention data
            if len(model['attention_usage']) > 100:
                model['attention_usage'] = model['attention_usage'][-100:]

    async def _discover_patterns(self, event: AMAPILearningEvent) -> List[Dict[str, Any]]:
        """Discover patterns from learning event"""
        patterns = []
        
        try:
            # Pattern discovery based on event type
            if event.event_type == LearningType.PATTERN_DISCOVERY:
                # Cross-reference with existing patterns
                new_pattern = event.learning_data.get('pattern', {})
                similar_patterns = self._find_similar_patterns(new_pattern)
                
                if similar_patterns:
                    # Merge or strengthen existing pattern
                    for similar in similar_patterns:
                        pattern = {
                            'type': 'pattern_reinforcement',
                            'original_pattern': similar,
                            'reinforcement_event': event.event_id,
                            'strength_increase': 0.1
                        }
                        patterns.append(pattern)
                else:
                    # New pattern discovered
                    pattern = {
                        'type': 'new_pattern',
                        'pattern_data': new_pattern,
                        'discovery_event': event.event_id,
                        'initial_strength': event.confidence
                    }
                    patterns.append(pattern)
            
            # Temporal pattern discovery
            temporal_patterns = await self._discover_temporal_patterns(event)
            patterns.extend(temporal_patterns)
            
            # Cross-agent pattern discovery
            if len(self.agent_behavioral_models) > 1:
                cross_agent_patterns = await self._discover_cross_agent_patterns(event)
                patterns.extend(cross_agent_patterns)
            
        except Exception as e:
            logger.error(f"Pattern discovery failed: {e}")
        
        return patterns

    async def _generate_predictions(self, event: AMAPILearningEvent) -> Dict[str, Any]:
        """Generate performance predictions based on learning"""
        predictions = {}
        
        try:
            agent_id = event.agent_id
            
            if agent_id in self.agent_behavioral_models:
                model = self.agent_behavioral_models[agent_id]
                
                # Predict success probability for similar tasks
                success_prediction = self._predict_task_success(model, event.learning_data)
                predictions['task_success_probability'] = success_prediction
                
                # Predict attention requirements
                attention_prediction = self._predict_attention_requirements(model, event.learning_data)
                predictions['attention_requirements'] = attention_prediction
                
                # Predict optimal collaboration partners
                collaboration_prediction = self._predict_collaboration_partners(agent_id, model)
                predictions['optimal_collaborators'] = collaboration_prediction
                
                # Predict potential failure points
                failure_prediction = self._predict_failure_points(model, event.learning_data)
                predictions['potential_failures'] = failure_prediction
                
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
        
        return predictions

    async def _facilitate_knowledge_transfer(self, event: AMAPILearningEvent) -> List[Dict[str, Any]]:
        """Facilitate knowledge transfer between agents"""
        transfers = []
        
        try:
            source_agent = event.agent_id
            learning_data = event.learning_data
            
            # Find agents that could benefit from this learning
            target_agents = self._identify_knowledge_transfer_targets(source_agent, learning_data)
            
            for target_agent in target_agents:
                transfer = {
                    'source_agent': source_agent,
                    'target_agent': target_agent,
                    'knowledge_type': event.event_type.value,
                    'knowledge_data': learning_data,
                    'transfer_confidence': self._calculate_transfer_confidence(source_agent, target_agent),
                    'expected_impact': self._calculate_expected_impact(target_agent, learning_data),
                    'timestamp': time.time()
                }
                
                transfers.append(transfer)
                
                # Update knowledge graph
                self.knowledge_graph[source_agent].append(target_agent)
                
        except Exception as e:
            logger.error(f"Knowledge transfer facilitation failed: {e}")
        
        return transfers

    def _calculate_impact_score(self, learning_data: Dict[str, Any]) -> float:
        """Calculate impact score for learning event"""
        base_score = 0.5
        
        # Factor in confidence
        confidence_factor = learning_data.get('confidence', 0.7) * 0.3
        
        # Factor in novelty
        novelty_factor = learning_data.get('novelty', 0.5) * 0.2
        
        # Factor in applicability
        applicability_factor = learning_data.get('applicability', 0.5) * 0.3
        
        # Factor in validation
        validation_factor = learning_data.get('validation_score', 0.5) * 0.2
        
        return min(1.0, base_score + confidence_factor + novelty_factor + applicability_factor + validation_factor)

    def _find_similar_patterns(self, new_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar patterns in global knowledge"""
        similar_patterns = []
        
        pattern_key = new_pattern.get('key', '')
        pattern_type = new_pattern.get('type', '')
        
        for global_pattern_key, global_pattern in self.global_patterns.items():
            if pattern_key in global_pattern_key or pattern_type == global_pattern.get('type'):
                similarity_score = self._calculate_pattern_similarity(new_pattern, global_pattern)
                if similarity_score > 0.7:
                    similar_patterns.append(global_pattern)
        
        return similar_patterns

    def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity score between two patterns"""
        # Simplified similarity calculation
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        total_keys = set(pattern1.keys()) | set(pattern2.keys())
        
        if not total_keys:
            return 0.0
        
        key_similarity = len(common_keys) / len(total_keys)
        
        # Value similarity for common keys
        value_similarity = 0.0
        if common_keys:
            for key in common_keys:
                if isinstance(pattern1[key], (int, float)) and isinstance(pattern2[key], (int, float)):
                    value_diff = abs(pattern1[key] - pattern2[key])
                    max_val = max(abs(pattern1[key]), abs(pattern2[key]), 1.0)
                    value_similarity += 1.0 - (value_diff / max_val)
                elif pattern1[key] == pattern2[key]:
                    value_similarity += 1.0
            
            value_similarity /= len(common_keys)
        
        return (key_similarity + value_similarity) / 2.0

    async def _discover_temporal_patterns(self, event: AMAPILearningEvent) -> List[Dict[str, Any]]:
        """Discover temporal patterns in learning events"""
        patterns = []
        
        # Get recent events from same agent
        agent_events = [e for e in self.learning_events[-50:] if e.agent_id == event.agent_id]
        
        if len(agent_events) >= 3:
            # Look for recurring sequences
            sequences = self._identify_event_sequences(agent_events)
            for sequence in sequences:
                if sequence['frequency'] >= 2:
                    pattern = {
                        'type': 'temporal_sequence',
                        'sequence': sequence,
                        'agent_id': event.agent_id,
                        'confidence': min(0.9, sequence['frequency'] / 10.0)
                    }
                    patterns.append(pattern)
        
        return patterns

    async def _discover_cross_agent_patterns(self, event: AMAPILearningEvent) -> List[Dict[str, Any]]:
        """Discover patterns across multiple agents"""
        patterns = []
        
        # Compare with other agents' recent patterns
        for other_agent_id, model in self.agent_behavioral_models.items():
            if other_agent_id != event.agent_id:
                similarity = self._calculate_behavioral_similarity(event.agent_id, other_agent_id)
                
                if similarity > 0.6:
                    pattern = {
                        'type': 'cross_agent_similarity',
                        'agents': [event.agent_id, other_agent_id],
                        'similarity_score': similarity,
                        'common_behaviors': self._identify_common_behaviors(event.agent_id, other_agent_id)
                    }
                    patterns.append(pattern)
        
        return patterns

    def _calculate_behavioral_similarity(self, agent1_id: str, agent2_id: str) -> float:
        """Calculate behavioral similarity between two agents"""
        if agent1_id not in self.agent_behavioral_models or agent2_id not in self.agent_behavioral_models:
            return 0.0
        
        model1 = self.agent_behavioral_models[agent1_id]
        model2 = self.agent_behavioral_models[agent2_id]
        
        # Compare learning patterns
        patterns1 = set(model1['learning_patterns'].keys())
        patterns2 = set(model2['learning_patterns'].keys())
        
        if not patterns1 and not patterns2:
            return 0.5
        
        pattern_overlap = len(patterns1 & patterns2) / len(patterns1 | patterns2) if patterns1 | patterns2 else 0
        
        # Compare success indicators
        success1 = set(model1['success_indicators'].keys())
        success2 = set(model2['success_indicators'].keys())
        
        success_overlap = len(success1 & success2) / len(success1 | success2) if success1 | success2 else 0
        
        # Compare adaptation rates
        adaptation_similarity = 1.0 - abs(model1['adaptation_rate'] - model2['adaptation_rate'])
        
        return (pattern_overlap + success_overlap + adaptation_similarity) / 3.0

    def _predict_task_success(self, model: Dict[str, Any], task_data: Dict[str, Any]) -> float:
        """Predict task success probability"""
        base_probability = 0.5
        
        # Factor in learning patterns
        task_type = task_data.get('task_type', 'unknown')
        if task_type in model['learning_patterns']:
            pattern = model['learning_patterns'][task_type]
            base_probability = pattern['success_rate']
        
        # Factor in recent performance
        if model['attention_usage']:
            recent_efficiency = np.mean([a['efficiency'] for a in model['attention_usage'][-10:]])
            base_probability = (base_probability + recent_efficiency) / 2
        
        # Factor in adaptation rate
        adaptation_bonus = min(0.2, model['adaptation_rate'] * 0.1)
        
        return min(0.95, base_probability + adaptation_bonus)

    def _predict_attention_requirements(self, model: Dict[str, Any], task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict attention requirements for task"""
        if not model['attention_usage']:
            return {'estimated_cost': 2.0, 'confidence': 0.3}
        
        recent_costs = [a['cost'] for a in model['attention_usage'][-20:]]
        avg_cost = np.mean(recent_costs)
        
        # Adjust based on task complexity
        complexity_factor = task_data.get('complexity', 1.0)
        estimated_cost = avg_cost * complexity_factor
        
        return {
            'estimated_cost': estimated_cost,
            'confidence': 0.7,
            'cost_range': (estimated_cost * 0.8, estimated_cost * 1.2)
        }

    def _predict_collaboration_partners(self, agent_id: str, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict optimal collaboration partners"""
        partners = []
        
        # Analyze knowledge graph
        if agent_id in self.knowledge_graph:
            for partner_id in self.knowledge_graph[agent_id]:
                if partner_id in self.agent_behavioral_models:
                    compatibility = self._calculate_collaboration_compatibility(agent_id, partner_id)
                    partners.append({
                        'agent_id': partner_id,
                        'compatibility_score': compatibility,
                        'collaboration_type': 'knowledge_sharing'
                    })
        
        # Sort by compatibility
        partners.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return partners[:3]  # Top 3 partners

    def _calculate_collaboration_compatibility(self, agent1_id: str, agent2_id: str) -> float:
        """Calculate collaboration compatibility between agents"""
        behavioral_similarity = self._calculate_behavioral_similarity(agent1_id, agent2_id)
        
        # Complementary skills factor (opposite is better for collaboration)
        complementary_factor = 1.0 - behavioral_similarity
        
        # Balance similarity and complementarity
        compatibility = (behavioral_similarity * 0.3 + complementary_factor * 0.7)
        
        return compatibility

    async def _update_system_metrics(self) -> None:
        """Update system-wide AMAPI metrics"""
        try:
            # System Intelligence Quotient (SIQ)
            if self.learning_events:
                pattern_accuracy = self._calculate_pattern_accuracy()
                prediction_accuracy = self._calculate_prediction_accuracy()
                adaptation_speed = self._calculate_adaptation_speed()
                
                self.system_metrics['system_intelligence_quotient'] = (
                    pattern_accuracy * prediction_accuracy * adaptation_speed
                ) ** (1/3)  # Geometric mean
            
            # Collaborative Efficiency Index (CEI)
            if len(self.agent_behavioral_models) > 1:
                learning_impact = self._calculate_cross_agent_learning_impact()
                sharing_rate = len(self.knowledge_graph) / max(1, len(self.agent_behavioral_models))
                collective_success = self._calculate_collective_success_rate()
                
                self.system_metrics['collaborative_efficiency_index'] = (
                    learning_impact * sharing_rate * collective_success
                )
            
            # Adaptive Resilience Score (ARS)
            difficulty_adjustment = self._calculate_difficulty_adjustment_accuracy()
            recovery_speed = self._calculate_recovery_speed()
            optimization_response = self._calculate_optimization_response_time()
            
            self.system_metrics['adaptive_resilience_score'] = (
                difficulty_adjustment * recovery_speed * optimization_response
            )
            
            # Predictive Precision Rating (PPR)
            success_prediction_accuracy = self._calculate_success_prediction_accuracy()
            bottleneck_detection_rate = self._calculate_bottleneck_detection_rate()
            optimization_impact = self._calculate_optimization_impact()
            
            self.system_metrics['predictive_precision_rating'] = (
                success_prediction_accuracy * bottleneck_detection_rate * optimization_impact
            )
            
        except Exception as e:
            logger.error(f"System metrics update failed: {e}")

    def _calculate_pattern_accuracy(self) -> float:
        """Calculate pattern recognition accuracy"""
        if not self.global_patterns:
            return 0.5
        
        # Simplified accuracy calculation
        validated_patterns = sum(1 for p in self.global_patterns.values() 
                               if p.get('validation_score', 0) > 0.7)
        return validated_patterns / len(self.global_patterns)

    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy"""
        if not self.performance_predictions:
            return 0.5
        
        # Simplified accuracy based on successful vs failed predictions
        total_predictions = self.system_metrics['successful_predictions'] + self.system_metrics['failed_predictions']
        if total_predictions == 0:
            return 0.5
        
        return self.system_metrics['successful_predictions'] / total_predictions

    def _calculate_adaptation_speed(self) -> float:
        """Calculate adaptation speed across agents"""
        if not self.agent_behavioral_models:
            return 0.5
        
        avg_adaptation_rate = np.mean([
            model['adaptation_rate'] for model in self.agent_behavioral_models.values()
        ])
        
        return min(1.0, avg_adaptation_rate * 2.0)  # Scale to 0-1

    # Analytics and reporting
    def get_amapi_analytics(self) -> Dict[str, Any]:
        """Get comprehensive AMAPI analytics"""
        return {
            'system_metrics': self.system_metrics.copy(),
            'learning_events_summary': {
                'total_events': len(self.learning_events),
                'events_by_type': self._analyze_events_by_type(),
                'recent_event_rate': self._calculate_recent_event_rate()
            },
            'knowledge_graph_analysis': {
                'total_connections': sum(len(connections) for connections in self.knowledge_graph.values()),
                'agent_connectivity': {agent: len(connections) for agent, connections in self.knowledge_graph.items()},
                'knowledge_flow_efficiency': self._calculate_knowledge_flow_efficiency()
            },
            'behavioral_model_summary': {
                'agents_modeled': len(self.agent_behavioral_models),
                'average_specialization': self._calculate_average_specialization(),
                'collaboration_network_density': self._calculate_collaboration_density()
            },
            'prediction_performance': {
                'active_predictions': len(self.performance_predictions),
                'prediction_categories': list(set([
                    list(pred.keys()) for pred in self.performance_predictions.values()
                ])),
                'average_prediction_confidence': self._calculate_average_prediction_confidence()
            }
        }

    def _analyze_events_by_type(self) -> Dict[str, int]:
        """Analyze learning events by type"""
        event_counts = defaultdict(int)
        for event in self.learning_events:
            event_counts[event.event_type.value] += 1
        return dict(event_counts)

    def _calculate_recent_event_rate(self) -> float:
        """Calculate recent event rate (events per minute in last hour)"""
        one_hour_ago = time.time() - 3600
        recent_events = [e for e in self.learning_events if e.timestamp > one_hour_ago]
        return len(recent_events) / 60.0  # Events per minute

    def _calculate_knowledge_flow_efficiency(self) -> float:
        """Calculate efficiency of knowledge flow in the system"""
        if not self.knowledge_graph:
            return 0.0
        
        total_agents = len(self.agent_behavioral_models)
        if total_agents < 2:
            return 0.0
        
        # Calculate connectivity ratio
        connected_agents = len(self.knowledge_graph)
        connectivity_ratio = connected_agents / total_agents
        
        # Calculate transfer efficiency
        total_transfers = self.system_metrics['knowledge_transfers']
        total_events = self.system_metrics['total_learning_events']
        transfer_efficiency = total_transfers / max(1, total_events)
        
        return (connectivity_ratio + transfer_efficiency) / 2.0


__all__ = [
    "AMAPICore",
    "AMAPILearningEvent", 
    "LearningType"
]