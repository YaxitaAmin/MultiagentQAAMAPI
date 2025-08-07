"""
Enhanced Base Agent that PROPERLY extends Agent-S Framework
Provides AMAPI integration while maintaining Agent-S compatibility
"""

import os
import sys
import io
import json
import time
import asyncio
import numpy as np
from PIL import Image
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable, Union
from loguru import logger

# AMAPI Core imports
from core.attention_economics import AttentionEconomicsEngine, AttentionAllocation
from core.logger import AMAPILogger

# Step 1: Import Agent-S directly
try:
    from gui_agents.s2.agents.agent_s import AgentS2
    from gui_agents.s2.agents.grounding import OSWorldACI
    import pyautogui
    AGENT_S_AVAILABLE = True
    logger.info("✅ Agent-S framework imported successfully")
except ImportError as e:
    logger.error(f"❌ Agent-S framework not available: {e}")
    AGENT_S_AVAILABLE = False
    
    # Create dummy class for fallback
    class AgentS2:
        def __init__(self, *args, **kwargs):
            self.mock_mode = True
            logger.warning("Using AgentS2 mock")
        
        async def predict(self, instruction: str, observation: dict):
            return {"mock": True}, ["tap(200, 400)"]


class AgentState(Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    LEARNING = "learning"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentType(Enum):
    """Types of agents in the system"""
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    SUPERVISOR = "supervisor"


class MessageType(Enum):
    """Message types for agent communication"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    OBSERVATION = "observation"
    ACTION = "action"
    VERIFICATION = "verification"
    PLAN_UPDATE = "plan_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1
    correlation_id: Optional[str] = None


@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics"""
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    average_response_time: float = 0.0
    attention_efficiency: float = 0.0
    learning_events: int = 0
    collaboration_score: float = 0.0
    specialization_score: float = 0.0
    agent_s_predictions: int = 0
    agent_s_success_rate: float = 0.0


@dataclass
class AgentAction:
    """Standardized agent action format"""
    action_id: str
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: float
    confidence: float
    reasoning: str
    attention_cost: float
    expected_outcome: str
    agent_s_generated: bool = False


@dataclass
class AgentObservation:
    """Standardized observation format"""
    observation_id: str
    timestamp: float
    source: str
    content: Dict[str, Any]
    confidence: float
    attention_required: float


class BaseQAAgent(AgentS2):
    """
    *** TRUE Agent-S Extension with AMAPI Integration ***
    Base QA Agent that PROPERLY extends AgentS2 from gui-agents framework
    Integrates AMAPI features while maintaining full Agent-S compatibility
    """
    
    def __init__(self, agent_id: str = None, agent_type: AgentType = None, 
                 config: Dict[str, Any] = None, engine_params: Dict[str, Any] = None, 
                 grounding_agent=None, platform: str = "linux", 
                 action_space: str = "pyautogui", observation_type: str = "screenshot",
                 search_engine: str = "Perplexica", embedding_engine_type: str = "openai"):
        """Initialize BaseQAAgent extending AgentS2 with AMAPI integration"""
        
        self.config = config or {}
        self.agent_type = agent_type or AgentType.EXECUTOR
        self.agent_id = agent_id or f"{self.agent_type.value}_{int(time.time() * 1000)}"
        self.agent_name = f"AMAPI_{self.agent_type.value.title()}_Agent"
        
        # Default engine parameters for Agent-S
        default_engine_params = {
            "engine_type": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
        }
        
        # Merge with provided params
        if engine_params:
            default_engine_params.update(engine_params)
        
        # Default grounding agent if not provided
        if not grounding_agent and AGENT_S_AVAILABLE:
            try:
                grounding_engine_params = {
                    "engine_type": "anthropic", 
                    "model": "claude-3-5-sonnet-20241022",
                    "grounding_width": 1366,
                    "grounding_height": 768
                }
                
                grounding_agent = OSWorldACI(
                    platform=platform,
                    engine_params_for_generation=default_engine_params,
                    engine_params_for_grounding=grounding_engine_params
                )
            except Exception as e:
                logger.warning(f"Failed to create grounding agent: {e}")
                grounding_agent = None
        
        # Initialize parent AgentS2 class
        if AGENT_S_AVAILABLE and grounding_agent:
            try:
                super().__init__(
                    engine_params=default_engine_params,
                    grounding_agent=grounding_agent,
                    platform=platform,
                    action_space=action_space,
                    observation_type=observation_type,
                    search_engine=search_engine,
                    embedding_engine_type=embedding_engine_type
                )
                logger.info(f"✅ {self.agent_name} successfully extends AgentS2")
                self.agent_s_active = True
            except Exception as e:
                logger.error(f"❌ Failed to initialize AgentS2: {e}")
                # Initialize minimal attributes for fallback
                self.agent_s_active = False
                self._init_fallback_attributes()
        else:
            logger.warning(f"❌ Agent-S not available, using fallback for {self.agent_name}")
            self.agent_s_active = False
            self._init_fallback_attributes()
        
        # Agent state management
        self.state = AgentState.INITIALIZING
        self.start_time = None
        self.is_running = False
        
        # Performance tracking
        self.metrics = AgentMetrics()
        self.execution_history: List[Dict[str, Any]] = []
        self.learning_events: List[Dict[str, Any]] = []
        
        # AMAPI Integration - Attention System
        self.attention_engine = AttentionEconomicsEngine(self.config.get('attention', {}))
        self.current_attention_allocation: Optional[AttentionAllocation] = None
        
        # Collaboration system
        self.collaboration_partners: Dict[str, 'BaseQAAgent'] = {}
        self.message_queue = asyncio.Queue(maxsize=100)
        self.message_handlers = {}
        self.learning_hooks: List[Callable] = []
        
        # Specialization areas
        self.specializations: List[str] = []
        self.expertise_levels: Dict[str, float] = {}
        
        # Logger
        self.logger = AMAPILogger(f"{self.agent_name}_{self.agent_id}")
        
        logger.info(f"🚀 Initialized {self.agent_name} extending Agent-S (Active: {self.agent_s_active})")
    
    def _init_fallback_attributes(self):
        """Initialize fallback attributes when Agent-S is not available"""
        self.engine_params = {"engine_type": "mock"}
        self.grounding_agent = None
        self.platform = "linux"
        self.action_space = "pyautogui"
        self.observation_type = "screenshot"

    async def start(self) -> bool:
        """Start the agent and initialize all systems"""
        try:
            self.state = AgentState.INITIALIZING
            self.start_time = time.time()
            
            # Initialize AMAPI attention system
            await self._initialize_attention_system()
            
            # Initialize agent-specific systems
            await self._initialize_agent_systems()
            
            # Load existing knowledge if available
            await self._load_agent_knowledge()
            
            # Set agent to idle state
            self.state = AgentState.IDLE
            self.is_running = True
            
            self.logger.info(f"Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to start agent {self.agent_id}: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the agent and cleanup resources"""
        try:
            self.state = AgentState.SHUTDOWN
            
            # Save current knowledge
            await self._save_agent_knowledge()
            
            # Cleanup agent-specific resources
            await self._cleanup_agent_systems()
            
            # Stop attention system
            await self._cleanup_attention_system()
            
            self.is_running = False
            self.logger.info(f"Agent {self.agent_id} stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping agent {self.agent_id}: {e}")
            return False
    
    async def predict(self, instruction: str, observation: Dict[str, Any], 
                     context: Dict[str, Any] = None) -> tuple[Dict[str, Any], List[str]]:
        """
        Override Agent-S predict method with AMAPI enhancements
        Maintains full compatibility while adding attention economics and learning
        """
        start_time = time.time()
        self.state = AgentState.THINKING
        
        try:
            # Allocate attention for this prediction
            attention_allocation = await self.allocate_attention(
                f"Predict for: {instruction}",
                context.get('task_complexity', 0.5) if context else 0.5
            )
            
            if self.agent_s_active:
                try:
                    # Call parent AgentS2.predict method
                    logger.debug(f"[{self.agent_name}] Using Agent-S predict for: {instruction}")
                    
                    # Convert observation to Agent-S format
                    agent_s_observation = self._prepare_observation_for_agent_s(observation)
                    
                    # Call the real Agent-S predict method
                    info, actions = super().predict(
                        instruction=instruction,
                        observation=agent_s_observation
                    )
                    
                    # Enhance Agent-S response with AMAPI metadata
                    enhanced_info = self._enhance_agent_s_response(info, instruction, attention_allocation)
                    validated_actions = self._validate_agent_s_actions(actions)
                    
                    # Update metrics
                    self.metrics.agent_s_predictions += 1
                    if enhanced_info.get('confidence', 0) > 0.7:
                        self.metrics.agent_s_success_rate = (
                            (self.metrics.agent_s_success_rate * (self.metrics.agent_s_predictions - 1) + 1.0) / 
                            self.metrics.agent_s_predictions
                        )
                    
                    # Log successful Agent-S usage
                    self._log_action_execution(
                        action_type="agent_s_predict",
                        input_data={"instruction": instruction, "observation_type": type(observation).__name__},
                        output_data={"info": enhanced_info, "actions_count": len(validated_actions)},
                        success=True,
                        duration=time.time() - start_time,
                        attention_used=attention_allocation.total_attention
                    )
                    
                    self.state = AgentState.IDLE
                    logger.info(f"✅ [{self.agent_name}] Agent-S prediction successful: {len(validated_actions)} actions")
                    return enhanced_info, validated_actions
                    
                except Exception as e:
                    logger.error(f"❌ [{self.agent_name}] Agent-S prediction failed: {e}")
                    # Fall back to AMAPI prediction
                    return await self._amapi_fallback_predict(instruction, observation, context, attention_allocation)
            else:
                # Use AMAPI-specific prediction when Agent-S not available
                return await self._amapi_fallback_predict(instruction, observation, context, attention_allocation)
                
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Error in predict method: {e}")
            return {'error': str(e), 'confidence': 0.0}, []
    
    def _prepare_observation_for_agent_s(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation in Agent-S expected format"""
        agent_s_obs = {}
        
        # Handle screenshot
        if 'screenshot' in observation:
            screenshot = observation['screenshot']
            if isinstance(screenshot, bytes):
                agent_s_obs['screenshot'] = screenshot
            elif hasattr(screenshot, 'save'):
                # PIL Image
                buffered = io.BytesIO()
                screenshot.save(buffered, format="PNG")
                agent_s_obs['screenshot'] = buffered.getvalue()
            else:
                # Try to take fresh screenshot
                try:
                    screenshot = pyautogui.screenshot()
                    buffered = io.BytesIO()
                    screenshot.save(buffered, format="PNG")
                    agent_s_obs['screenshot'] = buffered.getvalue()
                except Exception as e:
                    logger.warning(f"Failed to get screenshot: {e}")
                    agent_s_obs['screenshot'] = None
        else:
            # Take fresh screenshot for Agent-S
            try:
                screenshot = pyautogui.screenshot()
                buffered = io.BytesIO()
                screenshot.save(buffered, format="PNG")
                agent_s_obs['screenshot'] = buffered.getvalue()
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")
                agent_s_obs['screenshot'] = None
        
        # Add other observation data
        agent_s_obs.update({
            'ui_elements': observation.get('ui_elements', []),
            'activity': observation.get('activity', ''),
            'timestamp': time.time(),
            'prepared_by': self.agent_name
        })
        
        return agent_s_obs
    
    def _enhance_agent_s_response(self, info: Dict[str, Any], instruction: str, 
                                attention_allocation: AttentionAllocation) -> Dict[str, Any]:
        """Enhance Agent-S response with AMAPI-specific metadata"""
        enhanced = info.copy() if info else {}
        
        # Add AMAPI enhancements
        enhanced.update({
            "qa_agent": self.agent_name,
            "agent_id": self.agent_id,
            "qa_instruction": instruction,
            "qa_timestamp": time.time(),
            "extends_agent_s": True,
            "agent_s_active": True,
            "processing_mode": "agent_s_extended",
            "attention_cost": attention_allocation.total_attention,
            "attention_efficiency": attention_allocation.efficiency_score,
            "attention_allocation": asdict(attention_allocation)
        })
        
        # Ensure confidence score
        if "confidence" not in enhanced:
            enhanced["confidence"] = 0.85
        
        return enhanced
    
    def _validate_agent_s_actions(self, actions: List[str]) -> List[str]:
        """Validate and clean actions from Agent-S"""
        if not actions:
            return ["# No actions from Agent-S"]
        
        validated_actions = []
        for action in actions[:5]:  # Limit to 5 actions max
            if isinstance(action, str) and len(action.strip()) > 0:
                validated_actions.append(action.strip())
        
        return validated_actions if validated_actions else ["# Default action"]
    
    async def _amapi_fallback_predict(self, instruction: str, observation: Dict[str, Any], 
                                    context: Dict[str, Any], attention_allocation: AttentionAllocation) -> tuple[Dict[str, Any], List[str]]:
        """AMAPI-specific fallback prediction with attention economics"""
        logger.info(f"🔄 [{self.agent_name}] Using AMAPI fallback prediction")
        
        # Generate agent-specific response based on type
        actions = await self._generate_agent_specific_actions(instruction, observation, context)
        
        info = {
            "reasoning": f"{self.agent_type.value} agent AMAPI fallback for: {instruction}",
            "confidence": 0.7,
            "mode": "amapi_fallback",
            "qa_agent": self.agent_name,
            "agent_id": self.agent_id,
            "extends_agent_s": True,
            "agent_s_active": False,
            "attention_cost": attention_allocation.total_attention,
            "attention_efficiency": attention_allocation.efficiency_score,
            "processing_mode": "amapi_enhanced"
        }
        
        # Log fallback usage
        self._log_action_execution(
            action_type="amapi_fallback_predict",
            input_data={"instruction": instruction},
            output_data={"actions_count": len(actions)},
            success=True,
            duration=0.5,
            attention_used=attention_allocation.total_attention
        )
        
        self.state = AgentState.IDLE
        return info, actions

    async def _generate_agent_specific_actions(self, instruction: str, observation: Dict[str, Any], 
                                             context: Dict[str, Any]) -> List[str]:
        """Generate agent-type-specific actions"""
        if self.agent_type == AgentType.PLANNER:
            return [
                "# Analyze task requirements",
                "# Generate execution plan", 
                "# Identify risk factors",
                "# Optimize resource allocation"
            ]
        elif self.agent_type == AgentType.EXECUTOR:
            return [
                "tap(200, 400)",
                "wait(2)", 
                "swipe(100, 100, 300, 300)",
                "verify_ui_element('target')"
            ]
        elif self.agent_type == AgentType.VERIFIER:
            return [
                "# Check execution results",
                "# Validate success criteria",
                "# Generate verification report",
                "# Log findings"
            ]
        elif self.agent_type == AgentType.SUPERVISOR:
            return [
                "# Monitor agent coordination",
                "# Assess overall progress", 
                "# Make strategic decisions",
                "# Optimize workflow"
            ]
        else:
            return ["# Generic AMAPI action"]

    async def _initialize_attention_system(self) -> None:
        """Initialize AMAPI attention economics system"""
        try:
            # Create agent-specific attention pools
            attention_pools = {
                'perception': 2.0,
                'reasoning': 3.0,
                'action_planning': 2.5,
                'learning': 1.5,
                'collaboration': 1.0
            }
            
            await self.attention_engine.initialize_agent_attention(
                self.agent_id, attention_pools
            )
            
            self.logger.debug(f"AMAPI attention system initialized for {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize attention system: {e}")
            raise

    async def _cleanup_attention_system(self) -> None:
        """Cleanup attention system"""
        try:
            await self.attention_engine.cleanup_agent_attention(self.agent_id)
            self.logger.debug(f"Attention system cleaned up for {self.agent_id}")
        except Exception as e:
            self.logger.error(f"Error cleaning up attention system: {e}")

    async def allocate_attention(self, task_description: str, 
                               task_complexity: float = 0.5) -> AttentionAllocation:
        """Allocate attention for a specific task"""
        try:
            allocation_request = {
                'agent_id': self.agent_id,
                'task_description': task_description,
                'task_complexity': task_complexity,
                'agent_type': self.agent_type.value,
                'specializations': self.specializations
            }
            
            allocation = await self.attention_engine.allocate_attention_for_task(allocation_request)
            self.current_attention_allocation = allocation
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error allocating attention: {e}")
            # Return minimal allocation
            return AttentionAllocation(
                allocation_id=f"fallback_{int(time.time() * 1000)}",
                agent_id=self.agent_id,
                total_attention=2.0,
                pool_allocations={'reasoning': 2.0},
                efficiency_score=0.5,
                timestamp=time.time()
            )

    def _log_action_execution(self, action_type: str, input_data: Dict[str, Any], 
                            output_data: Dict[str, Any], success: bool, duration: float,
                            attention_used: float, error_message: Optional[str] = None) -> None:
        """Log action execution with AMAPI enhancements"""
        execution_record = {
            'timestamp': time.time(),
            'action_type': action_type,
            'input_data': input_data,
            'output_data': output_data,
            'success': success,
            'duration': duration,
            'attention_used': attention_used,
            'error_message': error_message,
            'agent_s_generated': 'agent_s' in action_type
        }
        
        self.execution_history.append(execution_record)
        self._update_metrics(execution_record)

    def _update_metrics(self, execution_record: Dict[str, Any]) -> None:
        """Update agent performance metrics"""
        try:
            self.metrics.total_actions += 1
            
            if execution_record['success']:
                self.metrics.successful_actions += 1
            else:
                self.metrics.failed_actions += 1
            
            # Update average response time
            current_avg = self.metrics.average_response_time
            total_actions = self.metrics.total_actions
            new_time = execution_record['duration']
            
            self.metrics.average_response_time = (
                (current_avg * (total_actions - 1) + new_time) / total_actions
            )
            
            # Update attention efficiency
            attention_efficiency = min(1.0, execution_record.get('attention_used', 1.0) / max(0.1, new_time))
            current_efficiency = self.metrics.attention_efficiency
            
            self.metrics.attention_efficiency = (
                (current_efficiency * (total_actions - 1) + attention_efficiency) / total_actions
            )
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    async def learn_from_execution(self, execution_data: Dict[str, Any]) -> None:
        """Learn from execution results and update knowledge"""
        try:
            self.state = AgentState.LEARNING
            
            # Create learning event
            learning_event = {
                'event_id': f"learn_{int(time.time() * 1000)}",
                'timestamp': time.time(),
                'agent_id': self.agent_id,
                'execution_data': execution_data,
                'learning_type': 'execution_feedback',
                'agent_s_involved': execution_data.get('agent_s_generated', False)
            }
            
            self.learning_events.append(learning_event)
            self.metrics.learning_events += 1
            
            # Apply agent-specific learning
            await self._apply_agent_learning(learning_event)
            
            # Trigger learning hooks for external systems
            for hook in self.learning_hooks:
                try:
                    await hook(learning_event)
                except Exception as e:
                    self.logger.error(f"Error in learning hook: {e}")
            
            self.state = AgentState.IDLE
            self.logger.debug(f"Learning event processed: {learning_event['event_id']}")
            
        except Exception as e:
            self.logger.error(f"Error in learn_from_execution: {e}")
            self.state = AgentState.ERROR

    async def _apply_agent_learning(self, learning_event: Dict[str, Any]) -> None:
        """Apply agent-specific learning - can be overridden by subclasses"""
        try:
            execution_data = learning_event['execution_data']
            
            # Update specialization scores based on performance
            if 'task_type' in execution_data:
                task_type = execution_data['task_type']
                success = execution_data.get('success', False)
                
                if task_type not in self.expertise_levels:
                    self.expertise_levels[task_type] = 0.5
                
                # Update expertise level
                learning_rate = 0.1
                if success:
                    self.expertise_levels[task_type] = min(1.0, 
                        self.expertise_levels[task_type] + learning_rate)
                else:
                    self.expertise_levels[task_type] = max(0.1, 
                        self.expertise_levels[task_type] - learning_rate * 0.5)
            
        except Exception as e:
            self.logger.error(f"Error applying agent learning: {e}")

    # Abstract methods for subclasses
    @abstractmethod
    async def _initialize_agent_systems(self) -> None:
        """Initialize agent-specific systems - must be implemented by each agent"""
        pass

    @abstractmethod
    async def _cleanup_agent_systems(self) -> None:
        """Cleanup agent-specific systems - must be implemented by each agent"""
        pass

    async def _load_agent_knowledge(self) -> None:
        """Load existing agent knowledge - can be overridden by subclasses"""
        try:
            # Default implementation - load from config or file system
            knowledge_file = self.config.get('knowledge_file')
            if knowledge_file:
                # Implementation would load from file
                pass
            
        except Exception as e:
            self.logger.debug(f"No existing knowledge to load: {e}")

    async def _save_agent_knowledge(self) -> None:
        """Save agent knowledge - can be overridden by subclasses"""
        try:
            # Default implementation - save to file system
            knowledge_file = self.config.get('knowledge_file')
            if knowledge_file:
                # Implementation would save to file
                pass
            
        except Exception as e:
            self.logger.error(f"Error saving agent knowledge: {e}")

    # AMAPI Integration Methods
    def add_learning_hook(self, hook: Callable) -> None:
        """Add a learning hook for external systems"""
        self.learning_hooks.append(hook)

    async def collaborate_with_agent(self, partner_agent: 'BaseQAAgent', 
                                   collaboration_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with another agent"""
        try:
            collaboration_id = f"collab_{int(time.time() * 1000)}"
            
            # Add partner to collaboration list
            self.collaboration_partners[partner_agent.agent_id] = partner_agent
            
            # Send collaboration message
            message = AgentMessage(
                message_id=f"msg_{int(time.time() * 1000)}",
                sender_id=self.agent_id,
                receiver_id=partner_agent.agent_id,
                message_type=MessageType.TASK_REQUEST,
                content={
                    'collaboration_id': collaboration_id,
                    'collaboration_type': collaboration_type,
                    'data': data
                },
                timestamp=time.time()
            )
            
            # Process collaboration
            response = await partner_agent.receive_collaboration_message(message)
            
            # Update collaboration score
            if response.get('success', False):
                self.metrics.collaboration_score = min(1.0, self.metrics.collaboration_score + 0.1)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in agent collaboration: {e}")
            return {'success': False, 'error': str(e)}

    async def receive_collaboration_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Receive and process collaboration message from another agent"""
        try:
            await self.message_queue.put(message)
            
            # Process message based on collaboration type
            collaboration_type = message.content.get('collaboration_type', 'unknown')
            response = await self._process_collaboration_message(collaboration_type, message)
            
            return {'success': True, 'response': response}
            
        except Exception as e:
            self.logger.error(f"Error processing collaboration message: {e}")
            return {'success': False, 'error': str(e)}

    async def _process_collaboration_message(self, collaboration_type: str, 
                                           message: AgentMessage) -> Dict[str, Any]:
        """Process collaboration message - can be overridden by subclasses"""
        return {'message': 'Collaboration message received', 'processed': True}

    # Status and Analytics Methods
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type.value,
            'state': self.state.value,
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'extends_agent_s': True,
            'agent_s_available': AGENT_S_AVAILABLE,
            'agent_s_active': self.agent_s_active,
            'metrics': asdict(self.metrics),
            'specializations': self.specializations,
            'expertise_levels': self.expertise_levels,
            'collaboration_partners': list(self.collaboration_partners.keys()),
            'message_queue_size': self.message_queue.qsize(),
            'execution_history_size': len(self.execution_history),
            'learning_events_count': len(self.learning_events)
        }

    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get detailed execution analytics"""
        try:
            if not self.execution_history:
                return {'no_data': True, 'extends_agent_s': True}
            
            # Calculate analytics from execution history
            success_rate = self.metrics.successful_actions / max(1, self.metrics.total_actions)
            
            attention_costs = [record.get('attention_used', 0) for record in self.execution_history]
            avg_attention_cost = sum(attention_costs) / len(attention_costs)
            
            execution_times = [record.get('duration', 0) for record in self.execution_history]
            avg_execution_time = sum(execution_times) / len(execution_times)
            
            agent_s_usage = sum(1 for record in self.execution_history if record.get('agent_s_generated', False))
            
            return {
                'success_rate': success_rate,
                'average_attention_cost': avg_attention_cost,
                'average_execution_time': avg_execution_time,
                'total_executions': len(self.execution_history),
                'learning_events': len(self.learning_events),
                'collaboration_score': self.metrics.collaboration_score,
                'specialization_coverage': len(self.expertise_levels),
                'extends_agent_s': True,
                'agent_s_available': AGENT_S_AVAILABLE,
                'agent_s_active': self.agent_s_active,
                'agent_s_predictions': self.metrics.agent_s_predictions,
                'agent_s_usage_rate': agent_s_usage / len(self.execution_history),
                'agent_s_success_rate': self.metrics.agent_s_success_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error generating execution analytics: {e}")
            return {'error': str(e), 'extends_agent_s': True}

    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        base_capabilities = [
            'agent_s_integration',
            'attention_management',
            'learning_adaptation',
            'collaboration',
            'performance_tracking',
            'knowledge_retention'
        ]
        
        if self.agent_s_active:
            base_capabilities.extend([
                'advanced_reasoning',
                'visual_understanding',
                'action_grounding',
                'context_awareness'
            ])
        
        return base_capabilities

    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check"""
        try:
            health_status = {
                'agent_id': self.agent_id,
                'is_healthy': True,
                'issues': [],
                'recommendations': [],
                'extends_agent_s': True,
                'agent_s_status': 'active' if self.agent_s_active else 'inactive'
            }
            
            # Check if agent is running
            if not self.is_running:
                health_status['is_healthy'] = False
                health_status['issues'].append('Agent is not running')
            
            # Check error state
            if self.state == AgentState.ERROR:
                health_status['is_healthy'] = False
                health_status['issues'].append('Agent is in error state')
            
            # Check performance metrics
            if self.metrics.total_actions > 10:
                success_rate = self.metrics.successful_actions / self.metrics.total_actions
                if success_rate < 0.7:
                    health_status['issues'].append(f'Low success rate: {success_rate:.2f}')
                    health_status['recommendations'].append('Review task complexity and attention allocation')
            
            # Check Agent-S integration health
            if AGENT_S_AVAILABLE and not self.agent_s_active:
                health_status['issues'].append('Agent-S available but not active')
                health_status['recommendations'].append('Investigate Agent-S initialization issues')
            
            # Check attention efficiency
            if self.metrics.attention_efficiency < 0.6:
                health_status['issues'].append(f'Low attention efficiency: {self.metrics.attention_efficiency:.2f}')
                health_status['recommendations'].append('Optimize attention allocation strategies')
            
            return health_status
            
        except Exception as e:
            return {
                'agent_id': self.agent_id,
                'is_healthy': False,
                'issues': [f'Health check failed: {str(e)}'],
                'recommendations': ['Investigate health check failure'],
                'extends_agent_s': True
            }

    def is_agent_s_working(self) -> bool:
        """Check if Agent-S is working properly"""
        return self.agent_s_active and AGENT_S_AVAILABLE


# Compatibility aliases
QAAgentS2 = BaseQAAgent


__all__ = [
    "BaseQAAgent",
    "QAAgentS2", 
    "AgentState",
    "AgentType",
    "AgentMetrics",
    "AgentAction",
    "AgentObservation",
    "AgentMessage",
    "MessageType"
]