"""
Pattern Recognition Engine for Multi-Agent QA System
Real-time pattern analysis and intelligent decision support
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import threading
from loguru import logger
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import cv2
import hashlib


@dataclass
class UIPattern:
    """Represents a recognized UI pattern"""
    pattern_id: str
    ui_signature: str
    screen_elements: List[Dict[str, Any]]
    action_sequence: List[str]
    success_contexts: List[Dict[str, Any]]
    failure_contexts: List[Dict[str, Any]]
    confidence_score: float
    recognition_count: int
    last_seen: float
    optimal_actions: Dict[str, Any]


@dataclass
class AttentionPattern:
    """Represents an attention allocation pattern"""
    pattern_id: str
    allocation_signature: str
    agent_distribution: Dict[str, float]
    task_context: Dict[str, Any]
    performance_outcome: Dict[str, float]
    efficiency_score: float
    usage_frequency: int
    optimization_potential: float
    timestamp: float


@dataclass
class ExecutionSequencePattern:
    """Represents a sequence execution pattern"""
    pattern_id: str
    sequence_signature: str
    agent_flow: List[str]
    decision_points: List[Dict[str, Any]]
    branching_logic: Dict[str, List[str]]
    success_probability: float
    average_completion_time: float
    critical_steps: List[int]
    failure_points: List[int]


class PatternRecognitionEngine:
    """
    Advanced pattern recognition system that identifies and learns from
    UI patterns, attention patterns, and execution sequence patterns
    """
    
    def __init__(self, learning_engine, persistence_path: str = "pattern_data"):
        """Initialize the pattern recognition engine"""
        self.learning_engine = learning_engine
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(exist_ok=True)
        
        # Pattern databases
        self.ui_patterns: Dict[str, UIPattern] = {}
        self.attention_patterns: Dict[str, AttentionPattern] = {}
        self.sequence_patterns: Dict[str, ExecutionSequencePattern] = {}
        
        # Recognition models
        self.ui_clusterer = DBSCAN(eps=0.3, min_samples=2)
        self.attention_clusterer = DBSCAN(eps=0.2, min_samples=3)
        self.sequence_analyzer = PCA(n_components=5)
        
        # Pattern analysis
        self.ui_features_history = deque(maxlen=500)
        self.attention_history = deque(maxlen=300)
        self.sequence_history = deque(maxlen=200)
        
        # Real-time recognition
        self.active_patterns: Set[str] = set()
        self.pattern_transitions: Dict[str, List[str]] = defaultdict(list)
        self.recognition_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.recognition_accuracy = 0.0
        self.pattern_prediction_success = 0.0
        self.optimization_impact = 0.0
        
        # Thread safety
        self.recognition_lock = threading.Lock()
        
        # Load existing patterns
        self._load_patterns()
        
        logger.info("PatternRecognitionEngine initialized")
    
    def analyze_current_state(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current execution state and recognize patterns
        Returns real-time pattern analysis and recommendations
        """
        with self.recognition_lock:
            try:
                analysis_start = time.time()
                
                # Extract features from current state
                ui_features = self._extract_ui_features(state_data)
                attention_features = self._extract_attention_features(state_data)
                sequence_features = self._extract_sequence_features(state_data)
                
                # Recognize patterns
                recognized_patterns = {
                    'ui_patterns': self._recognize_ui_patterns(ui_features),
                    'attention_patterns': self._recognize_attention_patterns(attention_features),
                    'sequence_patterns': self._recognize_sequence_patterns(sequence_features)
                }
                
                # Generate predictions
                predictions = self._generate_pattern_predictions(recognized_patterns, state_data)
                
                # Identify optimization opportunities
                optimizations = self._identify_optimization_opportunities(recognized_patterns, state_data)
                
                # Track pattern transitions
                self._track_pattern_transitions(recognized_patterns)
                
                # Update active patterns
                self._update_active_patterns(recognized_patterns)
                
                analysis_time = time.time() - analysis_start
                
                return {
                    'recognized_patterns': recognized_patterns,
                    'predictions': predictions,
                    'optimizations': optimizations,
                    'pattern_confidence': self._calculate_overall_confidence(recognized_patterns),
                    'analysis_time': analysis_time,
                    'active_pattern_count': len(self.active_patterns),
                    'recommendations': self._generate_real_time_recommendations(recognized_patterns),
                    'timestamp': time.time()
                }
                
            except Exception as e:
                logger.error(f"Error in pattern analysis: {e}")
                return {'error': str(e), 'timestamp': time.time()}
    
    def learn_from_execution_complete(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn patterns from completed execution
        Updates pattern databases with new insights
        """
        try:
            learning_impact = {'ui': 0.0, 'attention': 0.0, 'sequence': 0.0}
            
            # Learn UI patterns
            ui_pattern = self._learn_ui_pattern(execution_data)
            if ui_pattern:
                learning_impact['ui'] = self._update_ui_pattern_database(ui_pattern, execution_data)
            
            # Learn attention patterns
            attention_pattern = self._learn_attention_pattern(execution_data)
            if attention_pattern:
                learning_impact['attention'] = self._update_attention_pattern_database(attention_pattern, execution_data)
            
            # Learn sequence patterns
            sequence_pattern = self._learn_sequence_pattern(execution_data)
            if sequence_pattern:
                learning_impact['sequence'] = self._update_sequence_pattern_database(sequence_pattern, execution_data)
            
            # Update pattern relationships
            self._update_pattern_relationships(execution_data)
            
            # Optimize pattern database
            if len(self.ui_patterns) % 20 == 0:  # Periodic optimization
                self._optimize_pattern_database()
            
            return {
                'patterns_learned': {
                    'ui_pattern': ui_pattern.pattern_id if ui_pattern else None,
                    'attention_pattern': attention_pattern.pattern_id if attention_pattern else None,
                    'sequence_pattern': sequence_pattern.pattern_id if sequence_pattern else None
                },
                'learning_impact': learning_impact,
                'total_patterns': {
                    'ui': len(self.ui_patterns),
                    'attention': len(self.attention_patterns),
                    'sequence': len(self.sequence_patterns)
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error learning from execution: {e}")
            return {'error': str(e)}
    
    def predict_next_optimal_action(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the next optimal action based on recognized patterns
        """
        try:
            # Analyze current patterns
            current_analysis = self.analyze_current_state(current_context)
            recognized_patterns = current_analysis.get('recognized_patterns', {})
            
            # Find best matching patterns for prediction
            ui_matches = self._find_best_ui_matches(current_context)
            attention_matches = self._find_best_attention_matches(current_context)
            sequence_matches = self._find_best_sequence_matches(current_context)
            
            # Generate action predictions
            action_predictions = []
            
            # UI-based predictions
            for ui_pattern in ui_matches[:3]:  # Top 3 matches
                optimal_actions = ui_pattern.optimal_actions
                confidence = ui_pattern.confidence_score
                
                action_predictions.append({
                    'source': 'ui_pattern',
                    'pattern_id': ui_pattern.pattern_id,
                    'predicted_action': optimal_actions,
                    'confidence': confidence,
                    'success_rate': len(ui_pattern.success_contexts) / max(1, len(ui_pattern.success_contexts) + len(ui_pattern.failure_contexts))
                })
            
            # Attention-based predictions
            for attention_pattern in attention_matches[:2]:
                suggested_allocation = attention_pattern.agent_distribution
                efficiency = attention_pattern.efficiency_score
                
                action_predictions.append({
                    'source': 'attention_pattern',
                    'pattern_id': attention_pattern.pattern_id,
                    'predicted_action': {'attention_allocation': suggested_allocation},
                    'confidence': efficiency,
                    'efficiency_score': efficiency
                })
            
            # Sequence-based predictions
            for sequence_pattern in sequence_matches[:2]:
                next_agent = self._predict_next_agent_in_sequence(sequence_pattern, current_context)
                success_prob = sequence_pattern.success_probability
                
                action_predictions.append({
                    'source': 'sequence_pattern',
                    'pattern_id': sequence_pattern.pattern_id,
                    'predicted_action': {'next_agent': next_agent},
                    'confidence': success_prob,
                    'completion_time_estimate': sequence_pattern.average_completion_time
                })
            
            # Combine and rank predictions
            ranked_predictions = self._rank_action_predictions(action_predictions)
            
            # Generate final recommendation
            final_recommendation = self._generate_final_action_recommendation(ranked_predictions, current_context)
            
            return {
                'optimal_action': final_recommendation,
                'alternative_actions': ranked_predictions[:3],
                'prediction_confidence': self._calculate_prediction_confidence(ranked_predictions),
                'pattern_sources': [p['source'] for p in ranked_predictions[:3]],
                'estimated_success_rate': final_recommendation.get('estimated_success_rate', 0.5),
                'optimization_potential': self._calculate_optimization_potential(current_context),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error predicting optimal action: {e}")
            return {
                'optimal_action': {'action_type': 'continue', 'confidence': 0.1},
                'error': str(e)
            }
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive insights about recognized patterns
        """
        try:
            # UI pattern insights
            ui_insights = self._analyze_ui_pattern_trends()
            
            # Attention pattern insights
            attention_insights = self._analyze_attention_pattern_trends()
            
            # Sequence pattern insights
            sequence_insights = self._analyze_sequence_pattern_trends()
            
            # Cross-pattern correlations
            correlations = self._analyze_cross_pattern_correlations()
            
            # Performance metrics
            performance_metrics = {
                'recognition_accuracy': self.recognition_accuracy,
                'pattern_prediction_success': self.pattern_prediction_success,
                'optimization_impact': self.optimization_impact,
                'total_patterns_learned': len(self.ui_patterns) + len(self.attention_patterns) + len(self.sequence_patterns)
            }
            
            return {
                'ui_insights': ui_insights,
                'attention_insights': attention_insights,
                'sequence_insights': sequence_insights,
                'cross_pattern_correlations': correlations,
                'performance_metrics': performance_metrics,
                'pattern_database_stats': self._get_pattern_database_stats(),
                'learning_velocity': self._calculate_learning_velocity(),
                'pattern_evolution': self._track_pattern_evolution(),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating pattern insights: {e}")
            return {'error': str(e)}
    
    def _extract_ui_features(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract UI-related features for pattern recognition"""
        ui_features = {}
        
        # Basic UI structure
        ui_elements = state_data.get('ui_elements', [])
        ui_features['element_count'] = len(ui_elements)
        ui_features['element_types'] = [elem.get('class_name', 'unknown') for elem in ui_elements]
        ui_features['clickable_count'] = sum(1 for elem in ui_elements if elem.get('is_clickable', False))
        ui_features['editable_count'] = sum(1 for elem in ui_elements if elem.get('is_editable', False))
        
        # UI layout analysis
        ui_features['layout_signature'] = self._generate_layout_signature(ui_elements)
        ui_features['interaction_density'] = ui_features['clickable_count'] / max(1, ui_features['element_count'])
        
        # Screen content analysis
        ui_features['screen_context'] = self._analyze_screen_context(state_data)
        ui_features['app_context'] = state_data.get('current_app', 'unknown')
        
        # UI complexity metrics
        ui_features['complexity_score'] = self._calculate_ui_complexity(ui_elements)
        ui_features['navigation_depth'] = state_data.get('navigation_depth', 0)
        
        return ui_features
    
    def _extract_attention_features(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attention-related features for pattern recognition"""
        attention_features = {}
        
        # Current attention state
        attention_data = state_data.get('attention_usage', {})
        attention_features['current_allocation'] = attention_data
        attention_features['total_used'] = sum(attention_data.values())
        attention_features['allocation_entropy'] = self._calculate_allocation_entropy(attention_data)
        
        # Attention efficiency metrics
        steps_taken = state_data.get('steps_taken', 1)
        attention_features['efficiency'] = attention_features['total_used'] / steps_taken
        attention_features['distribution_balance'] = self._calculate_distribution_balance(attention_data)
        
        # Attention dynamics
        attention_features['agent_dominance'] = self._identify_dominant_agent(attention_data)
        attention_features['resource_pressure'] = self._calculate_resource_pressure(attention_data, state_data)
        
        return attention_features
    
    def _extract_sequence_features(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract execution sequence features for pattern recognition"""
        sequence_features = {}
        
        # Agent execution sequence
        agent_sequence = state_data.get('agent_sequence', [])
        sequence_features['agent_flow'] = agent_sequence
        sequence_features['sequence_length'] = len(agent_sequence)
        sequence_features['unique_agents'] = len(set(agent_sequence))
        
        # Sequence dynamics
        sequence_features['transition_patterns'] = self._analyze_agent_transitions(agent_sequence)
        sequence_features['coordination_complexity'] = len(agent_sequence) / max(1, len(set(agent_sequence)))
        
        # Decision points
        sequence_features['decision_points'] = state_data.get('decision_points', [])
        sequence_features['branching_factor'] = len(sequence_features['decision_points'])
        
        # Execution timing
        sequence_features['execution_timeline'] = state_data.get('step_timings', [])
        sequence_features['average_step_time'] = np.mean(sequence_features['execution_timeline']) if sequence_features['execution_timeline'] else 0
        
        return sequence_features
    
    def _recognize_ui_patterns(self, ui_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize UI patterns from current features"""
        recognized = []
        
        current_signature = ui_features.get('layout_signature', '')
        current_complexity = ui_features.get('complexity_score', 0)
        current_context = ui_features.get('app_context', 'unknown')
        
        # Find matching patterns
        for pattern_id, pattern in self.ui_patterns.items():
            similarity_score = self._calculate_ui_similarity(ui_features, pattern)
            
            if similarity_score > 0.7:  # High similarity threshold
                recognized.append({
                    'pattern_id': pattern_id,
                    'pattern': pattern,
                    'similarity_score': similarity_score,
                    'confidence': pattern.confidence_score,
                    'match_type': 'ui_layout'
                })
        
        # Sort by similarity and confidence
        recognized.sort(key=lambda x: x['similarity_score'] * x['confidence'], reverse=True)
        
        return recognized[:5]  # Return top 5 matches
    
    def _recognize_attention_patterns(self, attention_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize attention allocation patterns"""
        recognized = []
        
        current_allocation = attention_features.get('current_allocation', {})
        current_efficiency = attention_features.get('efficiency', 0)
        
        # Find matching attention patterns
        for pattern_id, pattern in self.attention_patterns.items():
            similarity_score = self._calculate_attention_similarity(attention_features, pattern)
            
            if similarity_score > 0.6:  # Attention pattern threshold
                recognized.append({
                    'pattern_id': pattern_id,
                    'pattern': pattern,
                    'similarity_score': similarity_score,
                    'efficiency_score': pattern.efficiency_score,
                    'match_type': 'attention_allocation'
                })
        
        # Sort by efficiency and similarity
        recognized.sort(key=lambda x: x['efficiency_score'] * x['similarity_score'], reverse=True)
        
        return recognized[:3]  # Return top 3 attention matches
    
    def _recognize_sequence_patterns(self, sequence_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize execution sequence patterns"""
        recognized = []
        
        current_flow = sequence_features.get('agent_flow', [])
        current_complexity = sequence_features.get('coordination_complexity', 1)
        
        # Find matching sequence patterns
        for pattern_id, pattern in self.sequence_patterns.items():
            similarity_score = self._calculate_sequence_similarity(sequence_features, pattern)
            
            if similarity_score > 0.5:  # Sequence pattern threshold
                recognized.append({
                    'pattern_id': pattern_id,
                    'pattern': pattern,
                    'similarity_score': similarity_score,
                    'success_probability': pattern.success_probability,
                    'match_type': 'execution_sequence'
                })
        
        # Sort by success probability and similarity
        recognized.sort(key=lambda x: x['success_probability'] * x['similarity_score'], reverse=True)
        
        return recognized[:3]  # Return top 3 sequence matches
    
    def _generate_layout_signature(self, ui_elements: List[Dict[str, Any]]) -> str:
        """Generate a unique signature for UI layout"""
        # Create signature based on element types and positions
        signature_elements = []
        
        for elem in sorted(ui_elements, key=lambda x: x.get('bbox_pixels', {}).get('y_min', 0)):
            elem_signature = f"{elem.get('class_name', 'unknown')}_{elem.get('is_clickable', False)}"
            signature_elements.append(elem_signature)
        
        # Create hash of signature
        signature_string = '_'.join(signature_elements[:20])  # Limit to first 20 elements
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]
    
    def _calculate_ui_complexity(self, ui_elements: List[Dict[str, Any]]) -> float:
        """Calculate UI complexity score"""
        if not ui_elements:
            return 0.0
        
        complexity = 0.0
        
        # Element count contribution
        complexity += min(0.4, len(ui_elements) / 50.0)
        
        # Interactive element ratio
        interactive_count = sum(1 for elem in ui_elements if elem.get('is_clickable', False) or elem.get('is_editable', False))
        complexity += min(0.3, interactive_count / len(ui_elements))
        
        # Layout depth (based on nested elements)
        max_depth = 0
        for elem in ui_elements:
            xpath = elem.get('xpath', '')
            depth = xpath.count('/') if xpath else 0
            max_depth = max(max_depth, depth)
        
        complexity += min(0.3, max_depth / 10.0)
        
        return min(1.0, complexity)
    
    def _analyze_screen_context(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze screen context for pattern recognition"""
        context = {}
        
        # App identification
        context['app_package'] = state_data.get('app_package', 'unknown')
        context['activity_name'] = state_data.get('activity_name', 'unknown')
        
        # Screen type classification
        ui_elements = state_data.get('ui_elements', [])
        text_elements = [elem.get('text', '') for elem in ui_elements if elem.get('text')]
        
        if any('settings' in text.lower() for text in text_elements):
            context['screen_type'] = 'settings'
        elif any('wifi' in text.lower() or 'network' in text.lower() for text in text_elements):
            context['screen_type'] = 'network'
        elif len([elem for elem in ui_elements if elem.get('is_editable')]) > 2:
            context['screen_type'] = 'form'
        else:
            context['screen_type'] = 'general'
        
        return context
    
    def _calculate_allocation_entropy(self, attention_data: Dict[str, float]) -> float:
        """Calculate entropy of attention allocation"""
        if not attention_data:
            return 0.0
        
        total = sum(attention_data.values())
        if total == 0:
            return 0.0
        
        probabilities = [usage / total for usage in attention_data.values()]
        entropy = -sum(p * np.log2(p + 1e-8) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(attention_data)) if len(attention_data) > 1 else 1
        return entropy / max_entropy
    
    def _calculate_distribution_balance(self, attention_data: Dict[str, float]) -> float:
        """Calculate how balanced the attention distribution is"""
        if not attention_data:
            return 0.0
        
        values = list(attention_data.values())
        if len(values) <= 1:
            return 1.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Perfect balance = low standard deviation
        balance_score = 1.0 - min(1.0, std_val / (mean_val + 1e-8))
        return balance_score
    
    def _identify_dominant_agent(self, attention_data: Dict[str, float]) -> str:
        """Identify which agent is using most attention"""
        if not attention_data:
            return 'none'
        
        return max(attention_data.items(), key=lambda x: x[1])[0]
    
    def _calculate_resource_pressure(self, attention_data: Dict[str, float], state_data: Dict[str, Any]) -> float:
        """Calculate current resource pressure on attention system"""
        total_used = sum(attention_data.values())
        total_available = state_data.get('total_attention_budget', 100)
        
        pressure = total_used / total_available if total_available > 0 else 0
        return min(1.0, pressure)
    
    def _analyze_agent_transitions(self, agent_sequence: List[str]) -> Dict[str, int]:
        """Analyze transitions between agents"""
        transitions = defaultdict(int)
        
        for i in range(len(agent_sequence) - 1):
            current_agent = agent_sequence[i]
            next_agent = agent_sequence[i + 1]
            transition = f"{current_agent}->{next_agent}"
            transitions[transition] += 1
        
        return dict(transitions)
    
    def _calculate_ui_similarity(self, ui_features: Dict[str, Any], pattern: UIPattern) -> float:
        """Calculate similarity between current UI and stored pattern"""
        similarity = 0.0
        
        # Layout signature match
        current_sig = ui_features.get('layout_signature', '')
        pattern_sig = pattern.ui_signature
        
        if current_sig == pattern_sig:
            similarity += 0.5  # Exact layout match
        
        # Element count similarity
        current_count = ui_features.get('element_count', 0)
        pattern_elements = len(pattern.screen_elements)
        
        if pattern_elements > 0:
            count_similarity = 1.0 - abs(current_count - pattern_elements) / max(current_count, pattern_elements)
            similarity += 0.2 * count_similarity
        
        # Complexity similarity
        current_complexity = ui_features.get('complexity_score', 0)
        pattern_complexity = sum(1 for elem in pattern.screen_elements if elem.get('is_clickable', False)) / max(1, len(pattern.screen_elements))
        
        complexity_similarity = 1.0 - abs(current_complexity - pattern_complexity)
        similarity += 0.3 * complexity_similarity
        
        return min(1.0, similarity)
    
    def _calculate_attention_similarity(self, attention_features: Dict[str, Any], pattern: AttentionPattern) -> float:
        """Calculate similarity between current attention and stored pattern"""
        current_allocation = attention_features.get('current_allocation', {})
        pattern_allocation = pattern.agent_distribution
        
        # Calculate cosine similarity of allocation vectors
        agents = set(current_allocation.keys()) | set(pattern_allocation.keys())
        current_vector = [current_allocation.get(agent, 0) for agent in agents]
        pattern_vector = [pattern_allocation.get(agent, 0) for agent in agents]
        
        if sum(current_vector) == 0 or sum(pattern_vector) == 0:
            return 0.0
        
        similarity = cosine_similarity([current_vector], [pattern_vector])[0][0]
        return max(0.0, similarity)
    
    def _calculate_sequence_similarity(self, sequence_features: Dict[str, Any], pattern: ExecutionSequencePattern) -> float:
        """Calculate similarity between current sequence and stored pattern"""
        current_flow = sequence_features.get('agent_flow', [])
        pattern_flow = pattern.agent_flow
        
        if not current_flow or not pattern_flow:
            return 0.0
        
        # Use sequence alignment similarity
        similarity = self._sequence_alignment_similarity(current_flow, pattern_flow)
        
        # Add coordination complexity similarity
        current_complexity = sequence_features.get('coordination_complexity', 1)
        pattern_complexity = len(pattern_flow) / max(1, len(set(pattern_flow)))
        
        complexity_similarity = 1.0 - abs(current_complexity - pattern_complexity) / max(current_complexity, pattern_complexity)
        
        return (similarity + complexity_similarity) / 2
    
    def _sequence_alignment_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences using alignment"""
        if not seq1 or not seq2:
            return 0.0
        
        # Simple longest common subsequence ratio
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        
        return lcs_len / max_len if max_len > 0 else 0.0
    
    def _learn_ui_pattern(self, execution_data: Dict[str, Any]) -> Optional[UIPattern]:
        """Learn UI pattern from execution data"""
        try:
            ui_states = execution_data.get('ui_states', [])
            if not ui_states:
                return None
            
            # Use the most complex UI state as representative
            representative_ui = max(ui_states, key=lambda state: len(state.get('ui_elements', [])))
            
            ui_signature = self._generate_layout_signature(representative_ui.get('ui_elements', []))
            
            # Create pattern
            pattern = UIPattern(
                pattern_id=f"ui_{ui_signature}_{int(time.time())}",
                ui_signature=ui_signature,
                screen_elements=representative_ui.get('ui_elements', []),
                action_sequence=execution_data.get('action_sequence', []),
                success_contexts=[execution_data] if execution_data.get('success', False) else [],
                failure_contexts=[execution_data] if not execution_data.get('success', False) else [],
                confidence_score=0.5,  # Initial confidence
                recognition_count=1,
                last_seen=time.time(),
                optimal_actions=self._extract_optimal_actions(execution_data)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error learning UI pattern: {e}")
            return None
    
    def _learn_attention_pattern(self, execution_data: Dict[str, Any]) -> Optional[AttentionPattern]:
        """Learn attention pattern from execution data"""
        try:
            attention_data = execution_data.get('attention_usage', {})
            if not attention_data:
                return None
            
            # Calculate performance outcome
            performance = {
                'success_rate': 1.0 if execution_data.get('success', False) else 0.0,
                'efficiency': execution_data.get('execution_time', 60) / max(1, execution_data.get('steps_taken', 1)),
                'completion_time': execution_data.get('execution_time', 60)
            }
            
            # Create allocation signature
            sorted_agents = sorted(attention_data.keys())
            allocation_values = [attention_data[agent] for agent in sorted_agents]
            allocation_signature = '_'.join([f"{agent}:{val:.2f}" for agent, val in zip(sorted_agents, allocation_values)])
            
            pattern = AttentionPattern(
                pattern_id=f"att_{hashlib.md5(allocation_signature.encode()).hexdigest()[:12]}",
                allocation_signature=allocation_signature,
                agent_distribution=attention_data,
                task_context={'task_name': execution_data.get('task_name', 'unknown')},
                performance_outcome=performance,
                efficiency_score=self._calculate_attention_efficiency(attention_data, performance),
                usage_frequency=1,
                optimization_potential=self._calculate_optimization_potential_for_attention(attention_data, performance),
                timestamp=time.time()
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error learning attention pattern: {e}")
            return None
    
    def _learn_sequence_pattern(self, execution_data: Dict[str, Any]) -> Optional[ExecutionSequencePattern]:
        """Learn execution sequence pattern from execution data"""
        try:
            agent_sequence = execution_data.get('agent_sequence', [])
            if not agent_sequence:
                return None
            
            # Create sequence signature
            sequence_signature = '_'.join(agent_sequence)
            
            # Identify decision points and critical steps
            decision_points = execution_data.get('decision_points', [])
            critical_steps = self._identify_critical_steps(execution_data)
            failure_points = self._identify_failure_points(execution_data)
            
            pattern = ExecutionSequencePattern(
                pattern_id=f"seq_{hashlib.md5(sequence_signature.encode()).hexdigest()[:12]}",
                sequence_signature=sequence_signature,
                agent_flow=agent_sequence,
                decision_points=decision_points,
                branching_logic=self._analyze_branching_logic(execution_data),
                success_probability=1.0 if execution_data.get('success', False) else 0.0,
                average_completion_time=execution_data.get('execution_time', 60),
                critical_steps=critical_steps,
                failure_points=failure_points
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error learning sequence pattern: {e}")
            return None
    
    def _extract_optimal_actions(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract optimal actions from successful execution"""
        if not execution_data.get('success', False):
            return {}
        
        actions = execution_data.get('actions', [])
        if not actions:
            return {}
        
        # Find the most effective action (one that led to success)
        optimal_action = actions[-1] if actions else {}
        
        return {
            'action_type': optimal_action.get('action_type', 'unknown'),
            'target_element': optimal_action.get('target_element', {}),
            'execution_time': optimal_action.get('execution_time', 0),
            'success_indicators': execution_data.get('success_indicators', [])
        }
    
    def _calculate_attention_efficiency(self, attention_data: Dict[str, float], performance: Dict[str, float]) -> float:
        """Calculate efficiency score for attention pattern"""
        total_attention = sum(attention_data.values())
        success_rate = performance.get('success_rate', 0)
        completion_time = performance.get('completion_time', 60)
        
        # Efficiency = success rate / (attention used * time taken)
        time_factor = min(1.0, 30.0 / completion_time)  # Normalize to 30 second baseline
        attention_factor = min(1.0, 50.0 / max(1, total_attention))  # Normalize to 50 attention baseline
        
        efficiency = success_rate * time_factor * attention_factor
        return min(1.0, efficiency)
    
    def _calculate_optimization_potential_for_attention(self, attention_data: Dict[str, float], performance: Dict[str, float]) -> float:
        """Calculate optimization potential for attention pattern"""
        # High potential if low efficiency or unbalanced allocation
        efficiency = self._calculate_attention_efficiency(attention_data, performance)
        balance = self._calculate_distribution_balance(attention_data)
        
        # Lower efficiency and balance = higher optimization potential
        potential = (1.0 - efficiency) * 0.7 + (1.0 - balance) * 0.3
        return min(1.0, potential)
    
    def _identify_critical_steps(self, execution_data: Dict[str, Any]) -> List[int]:
        """Identify critical steps in execution sequence"""
        critical_steps = []
        
        actions = execution_data.get('actions', [])
        step_timings = execution_data.get('step_timings', [])
        
        # Steps that took significantly longer are critical
        if step_timings:
            avg_time = np.mean(step_timings)
            std_time = np.std(step_timings)
            
            for i, timing in enumerate(step_timings):
                if timing > avg_time + std_time:  # Significantly longer than average
                    critical_steps.append(i)
        
        # Steps with high attention usage are critical
        attention_per_step = execution_data.get('attention_per_step', [])
        if attention_per_step:
            avg_attention = np.mean(attention_per_step)
            
            for i, attention in enumerate(attention_per_step):
                if attention > avg_attention * 1.5:  # 50% above average
                    critical_steps.append(i)
        
        return list(set(critical_steps))  # Remove duplicates
    
    def _identify_failure_points(self, execution_data: Dict[str, Any]) -> List[int]:
        """Identify potential failure points in execution sequence"""
        failure_points = []
        
        # Points where execution failed or nearly failed
        step_results = execution_data.get('step_results', [])
        for i, result in enumerate(step_results):
            if result.get('result') in ['failure', 'partial', 'retry_needed']:
                failure_points.append(i)
        
        # Points with high error rates
        error_indicators = execution_data.get('error_indicators', [])
        failure_points.extend(error_indicators)
        
        return list(set(failure_points))
    
    def _analyze_branching_logic(self, execution_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze branching logic in execution sequence"""
        branching_logic = defaultdict(list)
        
        agent_sequence = execution_data.get('agent_sequence', [])
        decision_points = execution_data.get('decision_points', [])
        
        # Map decision points to subsequent agent choices
        for decision_point in decision_points:
            step_index = decision_point.get('step_index', 0)
            if step_index < len(agent_sequence) - 1:
                current_agent = agent_sequence[step_index]
                next_agent = agent_sequence[step_index + 1]
                branching_logic[current_agent].append(next_agent)
        
        return dict(branching_logic)
    
    # Additional methods for pattern updates, predictions, and analysis would continue here...
    # This represents the core functionality of the Pattern Recognition Engine
    
    def _save_patterns(self):
        """Save all patterns to persistent storage"""
        try:
            import pickle
            
            with open(self.persistence_path / "ui_patterns.pkl", 'wb') as f:
                pickle.dump(self.ui_patterns, f)
            
            with open(self.persistence_path / "attention_patterns.pkl", 'wb') as f:
                pickle.dump(self.attention_patterns, f)
            
            with open(self.persistence_path / "sequence_patterns.pkl", 'wb') as f:
                pickle.dump(self.sequence_patterns, f)
                
            logger.info(f"Saved {len(self.ui_patterns)} UI patterns, {len(self.attention_patterns)} attention patterns, {len(self.sequence_patterns)} sequence patterns")
            
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def _load_patterns(self):
        """Load patterns from persistent storage"""
        try:
            import pickle
            
            ui_file = self.persistence_path / "ui_patterns.pkl"
            if ui_file.exists():
                with open(ui_file, 'rb') as f:
                    self.ui_patterns = pickle.load(f)
            
            attention_file = self.persistence_path / "attention_patterns.pkl"
            if attention_file.exists():
                with open(attention_file, 'rb') as f:
                    self.attention_patterns = pickle.load(f)
            
            sequence_file = self.persistence_path / "sequence_patterns.pkl"
            if sequence_file.exists():
                with open(sequence_file, 'rb') as f:
                    self.sequence_patterns = pickle.load(f)
                    
            logger.info(f"Loaded {len(self.ui_patterns)} UI patterns, {len(self.attention_patterns)} attention patterns, {len(self.sequence_patterns)} sequence patterns")
            
        except Exception as e:
            logger.warning(f"Could not load existing patterns: {e}")
