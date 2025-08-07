"""
Collaborative Knowledge Hub for AMAPI System
Manages cross-agent learning and knowledge sharing
"""

import time
import json
import uuid
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
from loguru import logger


class KnowledgeType(Enum):
    """Types of knowledge that can be shared"""
    PATTERN = "pattern"
    STRATEGY = "strategy"
    SOLUTION = "solution"
    ERROR_HANDLING = "error_handling"
    OPTIMIZATION = "optimization"
    EXPERIENCE = "experience"


@dataclass
class KnowledgeItem:
    """Individual piece of knowledge"""
    knowledge_id: str
    knowledge_type: KnowledgeType
    source_agent: str
    content: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float
    created_timestamp: float
    last_updated: float
    tags: List[str]
    context: Dict[str, Any]


@dataclass
class KnowledgeTransfer:
    """Record of knowledge transfer between agents"""
    transfer_id: str
    source_agent: str
    target_agent: str
    knowledge_item: KnowledgeItem
    transfer_timestamp: float
    transfer_success: bool
    impact_score: float
    feedback: Dict[str, Any]


@dataclass
class CollaborativeSession:
    """Collaborative learning session between agents"""
    session_id: str
    participating_agents: List[str]
    session_type: str
    shared_knowledge: List[str]  # Knowledge IDs
    outcomes: Dict[str, Any]
    start_time: float
    end_time: Optional[float]
    success_metrics: Dict[str, float]


class CollaborativeKnowledgeHub:
    """
    Central hub for collaborative knowledge management and sharing
    Implements cross-agent learning and collective intelligence
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Knowledge storage
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.knowledge_graph = nx.DiGraph()  # Knowledge relationships
        self.agent_expertise: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Transfer tracking
        self.transfer_history: List[KnowledgeTransfer] = []
        self.transfer_network = nx.DiGraph()  # Agent-to-agent transfers
        self.successful_transfers: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Collaborative sessions
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.session_history: List[CollaborativeSession] = []
        
        # Attention strategy distribution
        self.attention_strategies: Dict[str, Dict[str, Any]] = {}
        self.strategy_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.hub_metrics = {
            'total_knowledge_items': 0,
            'successful_transfers': 0,
            'active_collaborations': 0,
            'knowledge_utilization_rate': 0.0,
            'cross_agent_learning_impact': 0.0,
            'collective_intelligence_growth': 0.0,
            'attention_strategy_adoptions': 0
        }
        
        # Initialize knowledge categories
        self._initialize_knowledge_categories()
        
        logger.info("Collaborative Knowledge Hub initialized")

    def _initialize_knowledge_categories(self):
        """Initialize knowledge categorization system"""
        self.knowledge_categories = {
            KnowledgeType.PATTERN: {
                'description': 'Behavioral and execution patterns',
                'subcategories': ['success_patterns', 'failure_patterns', 'execution_patterns'],
                'sharing_weight': 0.8
            },
            KnowledgeType.STRATEGY: {
                'description': 'Task execution and attention strategies',
                'subcategories': ['attention_allocation', 'execution_strategies', 'planning_approaches'],
                'sharing_weight': 0.9
            },
            KnowledgeType.SOLUTION: {
                'description': 'Problem solutions and workarounds',
                'subcategories': ['bug_fixes', 'workarounds', 'optimizations'],
                'sharing_weight': 1.0
            },
            KnowledgeType.ERROR_HANDLING: {
                'description': 'Error recovery and handling techniques',
                'subcategories': ['error_patterns', 'recovery_strategies', 'prevention_methods'],
                'sharing_weight': 0.95
            },
            KnowledgeType.OPTIMIZATION: {
                'description': 'Performance optimizations and improvements',
                'subcategories': ['efficiency_improvements', 'resource_optimizations', 'speed_enhancements'],
                'sharing_weight': 0.85
            },
            KnowledgeType.EXPERIENCE: {
                'description': 'General experiences and insights',
                'subcategories': ['insights', 'lessons_learned', 'best_practices'],
                'sharing_weight': 0.7
            }
        }

    async def contribute_knowledge(self, agent_id: str, knowledge_type: KnowledgeType,
                                 content: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Agent contributes knowledge to the hub"""
        try:
            knowledge_id = f"knowledge_{uuid.uuid4().hex[:8]}"
            
            # Create knowledge item
            knowledge_item = KnowledgeItem(
                knowledge_id=knowledge_id,
                knowledge_type=knowledge_type,
                source_agent=agent_id,
                content=content,
                confidence=content.get('confidence', 0.7),
                usage_count=0,
                success_rate=1.0,  # Initial optimism
                created_timestamp=time.time(),
                last_updated=time.time(),
                tags=self._extract_tags(content),
                context=context or {}
            )
            
            # Store knowledge
            self.knowledge_base[knowledge_id] = knowledge_item
            
            # Update knowledge graph
            self._update_knowledge_graph(knowledge_item)
            
            # Update agent expertise
            self._update_agent_expertise(agent_id, knowledge_type, content)
            
            # Identify potential beneficiaries
            potential_beneficiaries = await self._identify_knowledge_beneficiaries(knowledge_item)
            
            # Initiate knowledge transfers
            transfers_initiated = 0
            for beneficiary in potential_beneficiaries:
                if beneficiary != agent_id:
                    transfer_success = await self._initiate_knowledge_transfer(
                        knowledge_item, beneficiary
                    )
                    if transfer_success:
                        transfers_initiated += 1
            
            # Update metrics
            self.hub_metrics['total_knowledge_items'] += 1
            
            logger.info(f"Knowledge contributed: {knowledge_type.value} from {agent_id}, "
                       f"initiated {transfers_initiated} transfers")
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error contributing knowledge: {e}")
            return ""

    def _extract_tags(self, content: Dict[str, Any]) -> List[str]:
        """Extract relevant tags from knowledge content"""
        tags = []
        
        # Extract from content keys
        for key in content.keys():
            if isinstance(key, str) and len(key) < 20:
                tags.append(key.lower())
        
        # Extract from text content
        text_content = str(content).lower()
        
        # Common QA-related tags
        qa_keywords = [
            'wifi', 'settings', 'app', 'launch', 'navigation', 'test',
            'success', 'failure', 'error', 'optimization', 'attention',
            'execution', 'verification', 'planning', 'strategy'
        ]
        
        for keyword in qa_keywords:
            if keyword in text_content:
                tags.append(keyword)
        
        # Remove duplicates and limit count
        return list(set(tags))[:10]

    def _update_knowledge_graph(self, knowledge_item: KnowledgeItem):
        """Update knowledge relationship graph"""
        knowledge_id = knowledge_item.knowledge_id
        
        # Add knowledge node
        self.knowledge_graph.add_node(
            knowledge_id,
            type=knowledge_item.knowledge_type.value,
            agent=knowledge_item.source_agent,
            confidence=knowledge_item.confidence,
            tags=knowledge_item.tags
        )
        
        # Find related knowledge and create edges
        for existing_id, existing_item in self.knowledge_base.items():
            if existing_id != knowledge_id:
                similarity = self._calculate_knowledge_similarity(knowledge_item, existing_item)
                if similarity > 0.7:  # High similarity threshold
                    self.knowledge_graph.add_edge(
                        knowledge_id, existing_id,
                        similarity=similarity,
                        relationship='similar'
                    )

    def _calculate_knowledge_similarity(self, item1: KnowledgeItem, item2: KnowledgeItem) -> float:
        """Calculate similarity between two knowledge items"""
        try:
            # Type similarity
            type_similarity = 1.0 if item1.knowledge_type == item2.knowledge_type else 0.3
            
            # Tag similarity
            tags1 = set(item1.tags)
            tags2 = set(item2.tags)
            tag_similarity = len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0
            
            # Content similarity (simplified)
            content1_str = str(item1.content).lower()
            content2_str = str(item2.content).lower()
            
            # Word overlap similarity
            words1 = set(content1_str.split())
            words2 = set(content2_str.split())
            word_similarity = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
            
            # Combine similarities
            overall_similarity = (
                type_similarity * 0.4 +
                tag_similarity * 0.3 +
                word_similarity * 0.3
            )
            
            return overall_similarity
            
        except Exception as e:
            logger.debug(f"Error calculating knowledge similarity: {e}")
            return 0.0

    def _update_agent_expertise(self, agent_id: str, knowledge_type: KnowledgeType, content: Dict[str, Any]):
        """Update agent expertise profile"""
        expertise_area = knowledge_type.value
        
        # Increase expertise in this area
        current_expertise = self.agent_expertise[agent_id][expertise_area]
        confidence = content.get('confidence', 0.7)
        
        # Weighted update
        weight = 0.1
        self.agent_expertise[agent_id][expertise_area] = (
            current_expertise * (1 - weight) + confidence * weight
        )
        
        # Update related areas based on content
        for tag in content.get('tags', []):
            if isinstance(tag, str):
                self.agent_expertise[agent_id][tag] = (
                    self.agent_expertise[agent_id][tag] * 0.95 + confidence * 0.05
                )

    async def _identify_knowledge_beneficiaries(self, knowledge_item: KnowledgeItem) -> List[str]:
        """Identify agents who would benefit from this knowledge"""
        beneficiaries = []
        
        try:
            knowledge_type = knowledge_item.knowledge_type
            tags = knowledge_item.tags
            source_agent = knowledge_item.source_agent
            
            # Check all agents
            for agent_id, expertise in self.agent_expertise.items():
                if agent_id == source_agent:
                    continue
                
                # Calculate benefit score
                benefit_score = 0.0
                
                # Expertise gap (agents with lower expertise benefit more)
                expertise_level = expertise.get(knowledge_type.value, 0.0)
                expertise_gap = max(0, knowledge_item.confidence - expertise_level)
                benefit_score += expertise_gap * 0.5
                
                # Tag relevance
                for tag in tags:
                    if expertise.get(tag, 0) < knowledge_item.confidence:
                        benefit_score += 0.1
                
                # Transfer success history
                transfer_key = (source_agent, agent_id)
                historical_success = self.successful_transfers.get(transfer_key, 0)
                if historical_success > 0:
                    benefit_score += min(0.2, historical_success * 0.05)
                
                # Add to beneficiaries if score is high enough
                if benefit_score > 0.3:
                    beneficiaries.append(agent_id)
            
            # Sort by benefit score (approximate)
            return beneficiaries[:5]  # Top 5 beneficiaries
            
        except Exception as e:
            logger.error(f"Error identifying knowledge beneficiaries: {e}")
            return []

    async def _initiate_knowledge_transfer(self, knowledge_item: KnowledgeItem, target_agent: str) -> bool:
        """Initiate knowledge transfer to target agent"""
        try:
            transfer_id = f"transfer_{uuid.uuid4().hex[:8]}"
            
            # Calculate transfer probability
            transfer_probability = self._calculate_transfer_probability(
                knowledge_item.source_agent, target_agent, knowledge_item
            )
            
            # Simulate transfer (in real implementation, this would notify the target agent)
            transfer_success = transfer_probability > 0.6
            
            # Create transfer record
            transfer = KnowledgeTransfer(
                transfer_id=transfer_id,
                source_agent=knowledge_item.source_agent,
                target_agent=target_agent,
                knowledge_item=knowledge_item,
                transfer_timestamp=time.time(),
                transfer_success=transfer_success,
                impact_score=transfer_probability if transfer_success else 0.0,
                feedback={}
            )
            
            # Store transfer
            self.transfer_history.append(transfer)
            
            # Update transfer network
            if not self.transfer_network.has_edge(knowledge_item.source_agent, target_agent):
                self.transfer_network.add_edge(knowledge_item.source_agent, target_agent, transfers=0)
            
            self.transfer_network[knowledge_item.source_agent][target_agent]['transfers'] += 1
            
            # Update success tracking
            if transfer_success:
                transfer_key = (knowledge_item.source_agent, target_agent)
                self.successful_transfers[transfer_key] += 1
                self.hub_metrics['successful_transfers'] += 1
                
                # Update knowledge usage
                knowledge_item.usage_count += 1
            
            return transfer_success
            
        except Exception as e:
            logger.error(f"Error initiating knowledge transfer: {e}")
            return False

    def _calculate_transfer_probability(self, source_agent: str, target_agent: str, 
                                      knowledge_item: KnowledgeItem) -> float:
        """Calculate probability of successful knowledge transfer"""
        try:
            probability = 0.5  # Base probability
            
            # Source expertise factor
            source_expertise = self.agent_expertise[source_agent].get(
                knowledge_item.knowledge_type.value, 0.5
            )
            probability += (source_expertise - 0.5) * 0.3
            
            # Knowledge confidence factor
            probability += (knowledge_item.confidence - 0.5) * 0.2
            
            # Historical transfer success
            transfer_key = (source_agent, target_agent)
            historical_success = self.successful_transfers.get(transfer_key, 0)
            if historical_success > 0:
                success_rate = historical_success / max(1, 
                    len([t for t in self.transfer_history 
                        if t.source_agent == source_agent and t.target_agent == target_agent])
                )
                probability += success_rate * 0.2
            
            # Knowledge category sharing weight
            category_weight = self.knowledge_categories[knowledge_item.knowledge_type]['sharing_weight']
            probability *= category_weight
            
            return max(0.1, min(0.95, probability))
            
        except Exception as e:
            logger.debug(f"Error calculating transfer probability: {e}")
            return 0.5

    async def start_collaborative_session(self, agent_ids: List[str], session_type: str) -> str:
        """Start collaborative learning session between agents"""
        try:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            
            session = CollaborativeSession(
                session_id=session_id,
                participating_agents=agent_ids,
                session_type=session_type,
                shared_knowledge=[],
                outcomes={},
                start_time=time.time(),
                end_time=None,
                success_metrics={}
            )
            
            self.active_sessions[session_id] = session
            self.hub_metrics['active_collaborations'] += 1
            
            # Identify relevant knowledge for sharing
            relevant_knowledge = await self._identify_session_knowledge(agent_ids, session_type)
            session.shared_knowledge = relevant_knowledge
            
            logger.info(f"Started collaborative session {session_id} with {len(agent_ids)} agents")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting collaborative session: {e}")
            return ""

    async def _identify_session_knowledge(self, agent_ids: List[str], session_type: str) -> List[str]:
        """Identify knowledge relevant to collaborative session"""
        relevant_knowledge = []
        
        try:
            # Get knowledge from participating agents
            agent_knowledge = {}
            for agent_id in agent_ids:
                agent_knowledge[agent_id] = [
                    item.knowledge_id for item in self.knowledge_base.values()
                    if item.source_agent == agent_id and item.success_rate > 0.7
                ]
            
            # Find complementary knowledge (what others don't have)
            for agent_id in agent_ids:
                for knowledge_id in agent_knowledge[agent_id]:
                    knowledge_item = self.knowledge_base[knowledge_id]
                    
                    # Check if other agents would benefit
                    would_benefit = False
                    for other_agent in agent_ids:
                        if other_agent != agent_id:
                            other_expertise = self.agent_expertise[other_agent].get(
                                knowledge_item.knowledge_type.value, 0.0
                            )
                            if other_expertise < knowledge_item.confidence - 0.2:
                                would_benefit = True
                                break
                    
                    if would_benefit:
                        relevant_knowledge.append(knowledge_id)
            
            return relevant_knowledge[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error identifying session knowledge: {e}")
            return []

    async def distribute_attention_strategy(self, source_agent: str, strategy: Dict[str, Any]) -> List[str]:
        """Distribute successful attention strategy to other agents"""
        try:
            strategy_id = f"strategy_{uuid.uuid4().hex[:8]}"
            
            # Store strategy
            self.attention_strategies[strategy_id] = {
                'source_agent': source_agent,
                'strategy': strategy,
                'effectiveness': strategy.get('effectiveness', 0.7),
                'created_timestamp': time.time(),
                'adoptions': 0
            }
            
            # Find agents who would benefit
            target_agents = []
            strategy_effectiveness = strategy.get('effectiveness', 0.7)
            
            for agent_id, expertise in self.agent_expertise.items():
                if agent_id != source_agent:
                    # Check if agent has lower effectiveness in relevant areas
                    current_effectiveness = expertise.get('attention_efficiency', 0.5)
                    if current_effectiveness < strategy_effectiveness - 0.1:
                        target_agents.append(agent_id)
            
            # Record adoptions
            adoption_count = 0
            for target_agent in target_agents:
                # Simulate strategy adoption (in real implementation, notify agent)
                adoption_probability = strategy_effectiveness * 0.8
                if np.random.random() < adoption_probability:
                    self._record_strategy_adoption(strategy_id, target_agent)
                    adoption_count += 1
            
            self.hub_metrics['attention_strategy_adoptions'] += adoption_count
            
            logger.info(f"Distributed attention strategy from {source_agent} to {adoption_count} agents")
            return target_agents[:adoption_count]
            
        except Exception as e:
            logger.error(f"Error distributing attention strategy: {e}")
            return []

    def _record_strategy_adoption(self, strategy_id: str, adopting_agent: str):
        """Record strategy adoption by agent"""
        if strategy_id in self.attention_strategies:
            self.attention_strategies[strategy_id]['adoptions'] += 1
            
            # Update strategy effectiveness tracking
            strategy_type = 'attention_strategy'  # Simplified
            if strategy_type not in self.strategy_effectiveness[adopting_agent]:
                self.strategy_effectiveness[adopting_agent][strategy_type] = []
            
            effectiveness = self.attention_strategies[strategy_id]['effectiveness']
            self.strategy_effectiveness[adopting_agent][strategy_type].append(effectiveness)

    async def query_knowledge(self, agent_id: str, query: Dict[str, Any]) -> List[KnowledgeItem]:
        """Query knowledge base for relevant items"""
        try:
            query_type = query.get('type')
            query_tags = query.get('tags', [])
            min_confidence = query.get('min_confidence', 0.5)
            
            relevant_items = []
            
            for knowledge_item in self.knowledge_base.values():
                # Skip own knowledge unless explicitly requested
                if knowledge_item.source_agent == agent_id and not query.get('include_own', False):
                    continue
                
                # Type matching
                if query_type and knowledge_item.knowledge_type.value != query_type:
                    continue
                
                # Confidence threshold
                if knowledge_item.confidence < min_confidence:
                    continue
                
                # Tag matching
                if query_tags:
                    tag_overlap = len(set(query_tags) & set(knowledge_item.tags))
                    if tag_overlap == 0:
                        continue
                
                relevant_items.append(knowledge_item)
            
            # Sort by relevance (confidence, usage, recency)
            relevant_items.sort(
                key=lambda item: (
                    item.confidence * 0.4 +
                    min(1.0, item.usage_count / 10.0) * 0.3 +
                    (1.0 - (time.time() - item.last_updated) / 86400) * 0.3
                ),
                reverse=True
            )
            
            return relevant_items[:10]  # Top 10 results
            
        except Exception as e:
            logger.error(f"Error querying knowledge: {e}")
            return []

    def calculate_collective_intelligence(self) -> float:
        """Calculate collective intelligence score of the system"""
        try:
            if not self.agent_expertise:
                return 0.0
            
            # Individual intelligence (average expertise across agents)
            individual_scores = []
            for agent_expertise in self.agent_expertise.values():
                if agent_expertise:
                    avg_expertise = np.mean(list(agent_expertise.values()))
                    individual_scores.append(avg_expertise)
            
            avg_individual = np.mean(individual_scores) if individual_scores else 0.0
            
            # Knowledge sharing factor
            total_knowledge = len(self.knowledge_base)
            successful_transfers = self.hub_metrics['successful_transfers']
            sharing_factor = min(1.0, successful_transfers / max(1, total_knowledge))
            
            # Collaboration factor
            completed_sessions = len(self.session_history)
            collaboration_factor = min(1.0, completed_sessions / 10.0)
            
            # Network effect (knowledge graph connectivity)
            if self.knowledge_graph.number_of_nodes() > 1:
                connectivity = (self.knowledge_graph.number_of_edges() / 
                              max(1, self.knowledge_graph.number_of_nodes()))
                network_factor = min(1.0, connectivity / 5.0)
            else:
                network_factor = 0.0
            
            # Combine factors
            collective_intelligence = (
                avg_individual * 0.4 +
                sharing_factor * 0.3 +
                collaboration_factor * 0.2 +
                network_factor * 0.1
            )
            
            return collective_intelligence
            
        except Exception as e:
            logger.error(f"Error calculating collective intelligence: {e}")
            return 0.0

    def get_hub_analytics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge hub analytics"""
        try:
            # Knowledge distribution
            knowledge_by_type = defaultdict(int)
            knowledge_by_agent = defaultdict(int)
            
            for item in self.knowledge_base.values():
                knowledge_by_type[item.knowledge_type.value] += 1
                knowledge_by_agent[item.source_agent] += 1
            
            # Transfer network analysis
            if self.transfer_network.number_of_nodes() > 0:
                network_density = nx.density(self.transfer_network)
                # Find most connected agents
                centrality = nx.degree_centrality(self.transfer_network)
                top_contributors = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            else:
                network_density = 0.0
                top_contributors = []
            
            # Recent activity
            recent_transfers = [t for t in self.transfer_history 
                             if time.time() - t.transfer_timestamp < 3600]  # Last hour
            
            # Strategy adoption analysis
            strategy_adoptions = sum(s['adoptions'] for s in self.attention_strategies.values())
            
            return {
                'hub_metrics': self.hub_metrics.copy(),
                'knowledge_distribution': {
                    'by_type': dict(knowledge_by_type),
                    'by_agent': dict(knowledge_by_agent),
                    'total_items': len(self.knowledge_base)
                },
                'transfer_analysis': {
                    'total_transfers': len(self.transfer_history),
                    'successful_transfers': self.hub_metrics['successful_transfers'],
                    'network_density': network_density,
                    'top_contributors': top_contributors,
                    'recent_transfers': len(recent_transfers)
                },
                'collaboration_metrics': {
                    'active_sessions': len(self.active_sessions),
                    'completed_sessions': len(self.session_history),
                    'total_strategies_shared': len(self.attention_strategies),
                    'strategy_adoptions': strategy_adoptions
                },
                'collective_intelligence': {
                    'score': self.calculate_collective_intelligence(),
                    'knowledge_graph_nodes': self.knowledge_graph.number_of_nodes(),
                    'knowledge_graph_edges': self.knowledge_graph.number_of_edges(),
                    'agent_count': len(self.agent_expertise)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating hub analytics: {e}")
            return {'error': str(e)}


__all__ = [
    "CollaborativeKnowledgeHub",
    "KnowledgeItem",
    "KnowledgeTransfer",
    "CollaborativeSession",
    "KnowledgeType"
]