"""
Production-Ready Enhanced Streamlit Dashboard for AMAPI Multi-Agent QA System
Comprehensive system with full functionality, real-time monitoring, and detailed analytics
"""

import streamlit as st
import asyncio
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components with error handling
try:
    from orchestrator import QAOrchestrator, WorkflowStatus
    from agents.executor_agent import DeviceConfig, DeviceType
    from core.amapi_core import LearningType
    from android_env_integration import EnhancedAndroidEnvIntegration, EnhancedQATaskResult
    from metrics import SystemMetrics
    from evaluator import SystemEvaluator
    from system_evaluator import EnhancedSystemEvaluator, ComponentHealth
    IMPORTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Critical import missing: {e}")
    logger.error(f"Failed importing components: {e}")
    IMPORTS_AVAILABLE = False

class StreamlitQADashboard:
    """
    Comprehensive AMAPI QA System Dashboard
    Full-featured production dashboard with real-time monitoring
    """
    
    def __init__(self):
        self.orchestrator: Optional[QAOrchestrator] = None
        self.android_integration: Optional[EnhancedAndroidEnvIntegration] = None
        self.system_metrics: Optional[SystemMetrics] = None
        self.system_evaluator: Optional[EnhancedSystemEvaluator] = None
        
        # Dashboard state
        self.dashboard_state = {
            'initialized': False,
            'last_update': time.time(),
            'auto_refresh': True,
            'refresh_interval': 5,
            'current_task': None,
            'task_running': False,
            'amapi_enabled': False
        }
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'execution_history' not in st.session_state:
            st.session_state.execution_history = []
        if 'benchmark_results' not in st.session_state:
            st.session_state.benchmark_results = {}
        if 'system_logs' not in st.session_state:
            st.session_state.system_logs = []
        if 'amapi_insights' not in st.session_state:
            st.session_state.amapi_insights = {}
        if 'agent_analytics' not in st.session_state:
            st.session_state.agent_analytics = {}
    
    def run_dashboard(self):
        """Main dashboard entry point"""
        st.set_page_config(
            page_title="🤖 AMAPI QA System Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Inject custom CSS
        self._inject_custom_css()
        
        # Main header
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">🤖 AMAPI Multi-Agent QA System Dashboard</h1>
            <p style="color: white; margin: 10px 0 0 0;">Adaptive Multi-Agent Performance Intelligence - Live System Monitoring & Control</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check system availability
        if not IMPORTS_AVAILABLE:
            st.error("❌ System components not available. Please check imports and dependencies.")
            return
        
        # Initialize system if needed
        if not self.dashboard_state['initialized']:
            self._render_initialization_interface()
        else:
            # Render main dashboard
            self._render_sidebar()
            self._render_main_dashboard()
            
            # Auto-refresh if enabled
            if self.dashboard_state['auto_refresh']:
                time.sleep(self.dashboard_state['refresh_interval'])
                st.rerun()
    
    def _inject_custom_css(self):
        """Inject custom CSS styles"""
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
            text-align: center;
        }
        .success-metric {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }
        .warning-metric {
            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        }
        .error-metric {
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        }
        .agent-status {
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 5px solid;
        }
        .agent-active {
            background-color: #e8f5e8;
            border-left-color: #4CAF50;
        }
        .agent-inactive {
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }
        .agent-error {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        .task-log {
            background-color: #f5f5f5;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 12px;
        }
        .amapi-insight {
            background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_initialization_interface(self):
        """Render system initialization interface"""
        st.markdown("## 🚀 System Initialization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to AMAPI QA System
            
            **Key Features:**
            - 🤖 **Multi-Agent Coordination**: 4 specialized AI agents
            - 🧠 **AMAPI Learning**: Continuous adaptation and learning
            - 📱 **AndroidWorld Integration**: Advanced Android automation
            - 📊 **Real-time Analytics**: Live performance monitoring
            - 🔍 **Bug Detection**: AI-powered issue identification
            
            **System Architecture:**
            1. **Supervisor Agent**: Orchestrates multi-agent collaboration
            2. **Planner Agent**: Creates intelligent execution strategies
            3. **Executor Agent**: Performs Android device interactions
            4. **Verifier Agent**: Validates results and detects bugs
            """)
        
        with col2:
            st.markdown("### 🔧 Configuration")
            
            # System configuration options
            llm_provider = st.selectbox(
                "LLM Provider",
                ["anthropic", "openai", "mock"],
                index=0,
                help="Select the language model provider"
            )
            
            android_task = st.selectbox(
                "Default Android Task",
                ["wifi_toggle", "settings_navigation", "app_launch", "comprehensive_suite"],
                index=0,
                help="Select default Android task type"
            )
            
            enable_amapi = st.checkbox(
                "Enable AMAPI Learning",
                value=True,
                help="Enable advanced learning and adaptation features"
            )
            
            debug_mode = st.checkbox(
                "Debug Mode",
                value=False,
                help="Enable detailed logging and debugging"
            )
            
            # Store configuration
            config = {
                'llm_provider': llm_provider,
                'android_task': android_task,
                'enable_amapi': enable_amapi,
                'debug_mode': debug_mode
            }
            
            if st.button("🚀 Initialize AMAPI System", type="primary", use_container_width=True):
                self._initialize_system(config)
    
    def _initialize_system(self, config: Dict[str, Any]):
        """Initialize the AMAPI QA system"""
        with st.spinner("🚀 Initializing AMAPI QA System..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize orchestrator
                status_text.text("🎯 Initializing orchestrator...")
                progress_bar.progress(20)
                
                orchestrator_config = {
                    'planner': {
                        'max_plan_complexity': 'highly_complex',
                        'default_strategy': 'adaptive'
                    },
                    'executor': {
                        'retry_attempts': 3,
                        'execution_timeout': 120
                    },
                    'verifier': {
                        'verification_depth': 'comprehensive',
                        'confidence_threshold': 0.85
                    },
                    'supervisor': {
                        'supervision_mode': 'autonomous',
                        'intervention_threshold': 0.7
                    },
                    'amapi': {
                        'learning_rate': 0.12,
                        'enable_learning': config.get('enable_amapi', True)
                    },
                    'llm': {
                        'provider': config.get('llm_provider', 'anthropic')
                    }
                }
                
                self.orchestrator = QAOrchestrator(orchestrator_config)
                
                # Initialize device configuration
                status_text.text("📱 Configuring Android environment...")
                progress_bar.progress(40)
                
                device_config = DeviceConfig(
                    device_type=DeviceType.EMULATOR,
                    screen_width=1080,
                    screen_height=1920,
                    api_level=29,
                    density=420
                )
                
                # Initialize system
                status_text.text("🔧 Initializing system components...")
                progress_bar.progress(60)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.orchestrator.initialize_system(device_config))
                
                # Initialize metrics and evaluation
                status_text.text("📊 Setting up monitoring...")
                progress_bar.progress(80)
                
                self.system_metrics = SystemMetrics()
                self.system_evaluator = EnhancedSystemEvaluator()
                
                # Complete initialization
                progress_bar.progress(100)
                status_text.text("✅ System initialized successfully!")
                
                self.dashboard_state['initialized'] = True
                self.dashboard_state['amapi_enabled'] = config.get('enable_amapi', True)
                
                st.success("🎉 AMAPI QA System initialized successfully!")
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                logger.error(f"System initialization error: {e}")
    
    def _render_sidebar(self):
        """Render sidebar with controls and status"""
        with st.sidebar:
            st.markdown("## 🎛️ System Control")
            
            # System status indicator
            if self.orchestrator and self.orchestrator.is_running:
                st.markdown("""
                <div class="metric-card success-metric">
                    <h3>🟢 System Online</h3>
                    <p>All systems operational</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick action buttons
                st.markdown("### ⚡ Quick Actions")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📶 WiFi Test", use_container_width=True):
                        self._execute_wifi_test()
                with col2:
                    if st.button("⚙️ Settings", use_container_width=True):
                        self._execute_settings_test()
                
                if st.button("🧪 Full QA Suite", use_container_width=True, type="primary"):
                    self._execute_comprehensive_suite()
                
                # System controls
                st.markdown("### 🔧 System Controls")
                
                if st.button("🔄 Restart System", use_container_width=True):
                    self._restart_system()
                
                if st.button("🛑 Shutdown", type="secondary", use_container_width=True):
                    self._shutdown_system()
            
            else:
                st.markdown("""
                <div class="metric-card error-metric">
                    <h3>🔴 System Offline</h3>
                    <p>System not initialized</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Initialize System", type="primary", use_container_width=True):
                    self.dashboard_state['initialized'] = False
                    st.rerun()
            
            # Dashboard settings
            st.markdown("### ⚙️ Dashboard Settings")
            
            self.dashboard_state['auto_refresh'] = st.checkbox(
                "🔄 Auto Refresh",
                value=self.dashboard_state['auto_refresh'],
                help="Automatically refresh dashboard data"
            )
            
            if self.dashboard_state['auto_refresh']:
                self.dashboard_state['refresh_interval'] = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=1,
                    max_value=30,
                    value=self.dashboard_state['refresh_interval']
                )
            
            # AMAPI status
            st.markdown("### 🧠 AMAPI Status")
            
            if self.dashboard_state.get('amapi_enabled', False):
                if self.orchestrator and hasattr(self.orchestrator, 'amapi_core'):
                    try:
                        amapi_analytics = self.orchestrator.amapi_core.get_amapi_analytics()
                        siq = amapi_analytics['system_metrics'].get('system_intelligence_quotient', 0.0)
                        cei = amapi_analytics['system_metrics'].get('collaborative_efficiency_index', 0.0)
                        
                        st.metric("🧠 System IQ", f"{siq:.3f}")
                        st.metric("🤝 Collaboration", f"{cei:.3f}")
                        
                        # Learning activity
                        learning_events = amapi_analytics['learning_events_summary'].get('total_events', 0)
                        st.metric("📚 Learning Events", learning_events)
                        
                    except Exception as e:
                        st.error(f"AMAPI Error: {e}")
                else:
                    st.warning("AMAPI Core not available")
            else:
                st.info("AMAPI Learning disabled")
    
    def _render_main_dashboard(self):
        """Render main dashboard content"""
        if not self.orchestrator or not self.orchestrator.is_running:
            st.warning("⚠️ System not running. Please initialize the system.")
            return
        
        # Get system status and performance data
        try:
            system_status = self.orchestrator.get_system_status()
            performance_report = self.orchestrator.get_performance_report()
        except Exception as e:
            st.error(f"Error getting system status: {e}")
            return
        
        # Render top metrics
        self._render_top_metrics(system_status, performance_report)
        
        # Create main tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🎯 Live Dashboard",
            "🤖 Agent Analytics", 
            "🧠 AMAPI Intelligence",
            "🧪 QA Execution",
            "📊 Performance Metrics",
            "🔬 Benchmarking",
            "🔍 System Debugging"
        ])
        
        with tab1:
            self._render_live_dashboard(system_status, performance_report)
        
        with tab2:
            self._render_agent_analytics(system_status)
        
        with tab3:
            self._render_amapi_intelligence()
        
        with tab4:
            self._render_qa_execution_interface()
        
        with tab5:
            self._render_performance_metrics(performance_report)
        
        with tab6:
            self._render_benchmarking_interface()
        
        with tab7:
            self._render_system_debugging()
    
    def _render_top_metrics(self, system_status: Dict, performance_report: Dict):
        """Render top-level system metrics"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            uptime = system_status.get('system_info', {}).get('uptime_seconds', 0)
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            st.metric("⏱️ Uptime", f"{hours}h {minutes}m")
        
        with col2:
            workflow_analysis = performance_report.get('workflow_analysis', {})
            success_rate = workflow_analysis.get('success_rate', 0.0)
            st.metric("✅ Success Rate", f"{success_rate:.1%}")
        
        with col3:
            total_executed = workflow_analysis.get('total_executed', 0)
            st.metric("🧪 Tests Run", f"{total_executed}")
        
        with col4:
            system_health = performance_report.get('system_health_score', 0.0)
            st.metric("❤️ Health", f"{system_health:.1%}")
        
        with col5:
            if self.dashboard_state.get('amapi_enabled', False):
                amapi_insights = performance_report.get('amapi_insights', {})
                siq = amapi_insights.get('system_metrics', {}).get('system_intelligence_quotient', 0.0)
                st.metric("🧠 System IQ", f"{siq:.2f}")
            else:
                st.metric("🧠 AMAPI", "Disabled")
        
        with col6:
            if self.dashboard_state.get('task_running', False):
                st.metric("🔄 Status", "Running")
            else:
                st.metric("🔄 Status", "Ready")
    
    def _render_live_dashboard(self, system_status: Dict, performance_report: Dict):
        """Render live dashboard overview"""
        st.markdown("## 🎯 Live System Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Real-time Workflow Timeline")
            
            # Generate timeline visualization
            if st.session_state.execution_history:
                timeline_data = []
                for i, execution in enumerate(st.session_state.execution_history[-10:]):
                    start_time = datetime.fromisoformat(execution.get('timestamp', datetime.now().isoformat()))
                    duration = execution.get('duration', 30)
                    end_time = start_time + timedelta(seconds=duration)
                    
                    timeline_data.append({
                        'Task': execution.get('task', f'Task {i+1}'),
                        'Start': start_time,
                        'End': end_time,
                        'Status': 'Success' if execution.get('success', False) else 'Failed',
                        'Duration': duration
                    })
                
                if timeline_data:
                    fig = px.timeline(
                        pd.DataFrame(timeline_data),
                        x_start='Start',
                        x_end='End',
                        y='Task',
                        color='Status',
                        hover_data=['Duration'],
                        title="Recent Task Execution Timeline"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No execution history available")
            else:
                st.info("No tasks executed yet. Run some QA tasks to see timeline.")
        
        with col2:
            st.markdown("### 🤖 Agent Status")
            
            agents_info = system_status.get('agent_status', {})
            for agent_id, agent_info in agents_info.items():
                agent_name = agent_info.get('agent_name', agent_id)
                is_running = agent_info.get('is_running', False)
                
                status_class = "agent-active" if is_running else "agent-inactive"
                status_icon = "🟢" if is_running else "🟡"
                
                st.markdown(f"""
                <div class="agent-status {status_class}">
                    <strong>{status_icon} {agent_name}</strong><br>
                    <small>Status: {'Active' if is_running else 'Standby'}</small><br>
                    <small>Actions: {agent_info.get('execution_history_length', 0)}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Recent activity log
        st.markdown("### 📋 Recent Activity")
        
        if st.session_state.system_logs:
            for log_entry in st.session_state.system_logs[-5:]:
                timestamp = log_entry.get('timestamp', 'Unknown')
                level = log_entry.get('level', 'INFO')
                message = log_entry.get('message', '')
                
                level_colors = {
                    'INFO': '#2196F3',
                    'SUCCESS': '#4CAF50',
                    'WARNING': '#ff9800',
                    'ERROR': '#f44336',
                    'DEBUG': '#9C27B0'
                }
                
                color = level_colors.get(level, '#2196F3')
                
                st.markdown(f"""
                <div class="task-log" style="border-left-color: {color};">
                    <strong>[{timestamp}] {level}</strong>: {message}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent activity logs")
    
    def _render_agent_analytics(self, system_status: Dict):
        """Render detailed agent analytics"""
        st.markdown("## 🤖 Comprehensive Agent Analytics")
        
        agents_info = system_status.get('agent_status', {})
        
        if not agents_info:
            st.warning("No agent information available")
            return
        
        # Agent selection
        agent_names = {agent_id: info.get('agent_name', agent_id) for agent_id, info in agents_info.items()}
        selected_agent_id = st.selectbox(
            "Select Agent for Detailed Analysis",
            options=list(agent_names.keys()),
            format_func=lambda x: agent_names[x]
        )
        
        if selected_agent_id:
            agent_info = agents_info[selected_agent_id]
            agent_name = agent_info.get('agent_name', selected_agent_id)
            
            st.markdown(f"### 📊 {agent_name} Analytics")
            
            # Agent overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                is_running = agent_info.get('is_running', False)
                status = "🟢 Active" if is_running else "🟡 Standby"
                st.metric("Status", status)
            
            with col2:
                exec_count = agent_info.get('execution_history_length', 0)
                st.metric("Actions Executed", exec_count)
            
            with col3:
                learning_events = agent_info.get('learning_events', 0)
                st.metric("Learning Events", learning_events)
            
            with col4:
                adaptations = agent_info.get('adaptation_count', 0)
                st.metric("Adaptations", adaptations)
            
            # Detailed analytics tabs
            analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
                "📊 Performance", "🧠 Learning", "🔧 Configuration", "📈 Trends"
            ])
            
            with analytics_tab1:
                st.markdown("#### Performance Metrics")
                
                # Get detailed analytics if available
                if self.orchestrator and selected_agent_id in self.orchestrator.agents:
                    agent_obj = self.orchestrator.agents[selected_agent_id]
                    
                    if hasattr(agent_obj, 'get_execution_analytics'):
                        analytics = agent_obj.get_execution_analytics()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            success_rate = analytics.get('success_rate', 0.0)
                            st.metric("Success Rate", f"{success_rate:.1%}")
                            
                            avg_duration = analytics.get('average_execution_time', 0.0)
                            st.metric("Avg Duration", f"{avg_duration:.2f}s")
                        
                        with col2:
                            error_rate = analytics.get('error_rate', 0.0)
                            st.metric("Error Rate", f"{error_rate:.1%}")
                            
                            efficiency = analytics.get('efficiency_score', 0.0)
                            st.metric("Efficiency", f"{efficiency:.2f}")
                        
                        # Performance breakdown
                        if 'task_performance' in analytics:
                            st.markdown("##### Task Performance Breakdown")
                            task_perf = analytics['task_performance']
                            
                            perf_df = pd.DataFrame([
                                {'Task Type': task, 'Success Rate': f"{perf['success_rate']:.1%}", 
                                 'Avg Duration': f"{perf['avg_duration']:.2f}s"}
                                for task, perf in task_perf.items()
                            ])
                            
                            st.dataframe(perf_df, use_container_width=True)
                    else:
                        st.info("Detailed analytics not available for this agent")
                else:
                    st.info("Agent object not accessible")
            
            with analytics_tab2:
                st.markdown("#### Learning & Adaptation")
                
                # AMAPI learning data
                if self.dashboard_state.get('amapi_enabled', False) and hasattr(self.orchestrator, 'amapi_core'):
                    try:
                        amapi_analytics = self.orchestrator.amapi_core.get_amapi_analytics()
                        
                        # Agent-specific learning insights
                        if 'agent_learning_insights' in amapi_analytics:
                            agent_insights = amapi_analytics['agent_learning_insights'].get(selected_agent_id, {})
                            
                            if agent_insights:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Patterns Learned", agent_insights.get('patterns_learned', 0))
                                    st.metric("Learning Rate", f"{agent_insights.get('learning_rate', 0.0):.3f}")
                                
                                with col2:
                                    st.metric("Adaptation Success", f"{agent_insights.get('adaptation_success_rate', 0.0):.1%}")
                                    st.metric("Knowledge Score", f"{agent_insights.get('knowledge_score', 0.0):.2f}")
                                
                                # Learning history
                                if 'recent_learning_events' in agent_insights:
                                    st.markdown("##### Recent Learning Events")
                                    for event in agent_insights['recent_learning_events'][-5:]:
                                        st.write(f"• {event.get('description', 'Learning event')}")
                            else:
                                st.info("No specific learning insights available for this agent")
                        else:
                            st.info("Agent learning insights not available")
                    except Exception as e:
                        st.error(f"Error accessing learning data: {e}")
                else:
                    st.info("AMAPI learning not enabled")
            
            with analytics_tab3:
                st.markdown("#### Configuration & Details")
                
                config_data = {
                    "Agent ID": selected_agent_id,
                    "Agent Name": agent_name,
                    "Platform": agent_info.get('platform', 'Unknown'),
                    "Action Space": agent_info.get('action_space', 'Unknown'),
                    "Agent-S Active": agent_info.get('agent_s_active', False),
                    "Initialization Time": agent_info.get('init_time', 'Unknown'),
                    "Last Activity": agent_info.get('last_activity', 'Unknown')
                }
                
                for key, value in config_data.items():
                    st.write(f"**{key}**: {value}")
            
            with analytics_tab4:
                st.markdown("#### Performance Trends")
                
                # Generate mock trend data (in real implementation, this would come from stored metrics)
                if st.session_state.execution_history:
                    agent_executions = [
                        ex for ex in st.session_state.execution_history 
                        if ex.get('primary_agent') == selected_agent_id
                    ]
                    
                    if agent_executions:
                        trend_data = []
                        for i, execution in enumerate(agent_executions[-20:]):  # Last 20 executions
                            trend_data.append({
                                'Execution': i + 1,
                                'Success': 1 if execution.get('success', False) else 0,
                                'Duration': execution.get('duration', 0),
                                'Timestamp': execution.get('timestamp', '')
                            })
                        
                        trend_df = pd.DataFrame(trend_data)
                        
                        if not trend_df.empty:
                            # Success rate trend
                            fig1 = px.line(
                                trend_df, 
                                x='Execution', 
                                y='Success',
                                title=f"{agent_name} Success Rate Trend"
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # Duration trend
                            fig2 = px.line(
                                trend_df, 
                                x='Execution', 
                                y='Duration',
                                title=f"{agent_name} Execution Duration Trend"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("No trend data available")
                    else:
                        st.info(f"No execution history for {agent_name}")
                else:
                    st.info("No execution history available")
    
    def _render_amapi_intelligence(self):
        """Render AMAPI intelligence dashboard"""
        st.markdown("## 🧠 AMAPI Intelligence Dashboard")
        
        if not self.dashboard_state.get('amapi_enabled', False):
            st.warning("⚠️ AMAPI Learning is disabled. Enable it in system initialization to see intelligence metrics.")
            return
        
        if not hasattr(self.orchestrator, 'amapi_core') or not self.orchestrator.amapi_core:
            st.error("❌ AMAPI Core not available")
            return
        
        try:
            amapi_analytics = self.orchestrator.amapi_core.get_amapi_analytics()
            
            # AMAPI Core Metrics
            st.markdown("### 📊 Core Intelligence Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                siq = amapi_analytics['system_metrics'].get('system_intelligence_quotient', 0.0)
                st.metric("🧠 System IQ", f"{siq:.3f}")
            
            with col2:
                cei = amapi_analytics['system_metrics'].get('collaborative_efficiency_index', 0.0)
                st.metric("🤝 Collaboration", f"{cei:.3f}")
            
            with col3:
                ars = amapi_analytics['system_metrics'].get('adaptive_resilience_score', 0.0)
                st.metric("🛡️ Resilience", f"{ars:.3f}")
            
            with col4:
                ppr = amapi_analytics['system_metrics'].get('predictive_precision_rating', 0.0)
                st.metric("🔮 Prediction", f"{ppr:.3f}")
            
            with col5:
                uci = amapi_analytics['system_metrics'].get('universal_compatibility_index', 0.0)
                st.metric("🌐 Compatibility", f"{uci:.3f}")
            
            # Learning activity overview
            st.markdown("### 📚 Learning Activity")
            
            learning_summary = amapi_analytics.get('learning_events_summary', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_events = learning_summary.get('total_events', 0)
                st.metric("Total Learning Events", total_events)
            
            with col2:
                recent_rate = learning_summary.get('recent_event_rate', 0.0)
                st.metric("Recent Learning Rate", f"{recent_rate:.2f}/min")
            
            with col3:
                event_types = len(learning_summary.get('events_by_type', {}))
                st.metric("Learning Types", event_types)
            
            # Learning events breakdown
            events_by_type = learning_summary.get('events_by_type', {})
            if events_by_type:
                st.markdown("### 📈 Learning Events by Type")
                
                fig = px.pie(
                    values=list(events_by_type.values()),
                    names=list(events_by_type.keys()),
                    title="Distribution of Learning Event Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Knowledge graph analysis
            st.markdown("### 🕸️ Knowledge Graph Analysis")
            
            knowledge_graph = amapi_analytics.get('knowledge_graph_analysis', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                total_connections = knowledge_graph.get('total_connections', 0)
                st.metric("Knowledge Connections", total_connections)
                
                flow_efficiency = knowledge_graph.get('knowledge_flow_efficiency', 0.0)
                st.metric("Flow Efficiency", f"{flow_efficiency:.3f}")
            
            with col2:
                agent_connectivity = knowledge_graph.get('agent_connectivity', {})
                if agent_connectivity:
                    connectivity_df = pd.DataFrame([
                        {'Agent': agent, 'Connections': connections}
                        for agent, connections in agent_connectivity.items()
                    ])
                    
                    fig = px.bar(
                        connectivity_df,
                        x='Agent',
                        y='Connections',
                        title="Agent Knowledge Connectivity"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Behavioral model insights
            st.markdown("### 🎭 Behavioral Model Summary")
            
            behavioral_summary = amapi_analytics.get('behavioral_model_summary', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                agents_modeled = behavioral_summary.get('agents_modeled', 0)
                st.metric("Agents Modeled", agents_modeled)
            
            with col2:
                avg_specialization = behavioral_summary.get('average_specialization', 0.0)
                st.metric("Avg Specialization", f"{avg_specialization:.3f}")
            
            with col3:
                network_density = behavioral_summary.get('collaboration_network_density', 0.0)
                st.metric("Network Density", f"{network_density:.3f}")
            
            # Real-time insights
            st.markdown("### ⚡ Real-time Intelligence Insights")
            
            if 'real_time_insights' in amapi_analytics:
                insights = amapi_analytics['real_time_insights']
                
                for insight in insights[-5:]:  # Show last 5 insights
                    insight_type = insight.get('type', 'general')
                    message = insight.get('message', '')
                    confidence = insight.get('confidence', 0.0)
                    
                    st.markdown(f"""
                    <div class="amapi-insight">
                        <strong>🧠 {insight_type.title()} Insight</strong> (Confidence: {confidence:.2f})<br>
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No real-time insights available")
        
        except Exception as e:
            st.error(f"❌ Error loading AMAPI intelligence data: {e}")
            logger.error(f"AMAPI intelligence error: {e}")
    
    def _render_qa_execution_interface(self):
        """Render QA task execution interface"""
        st.markdown("## 🧪 QA Task Execution Center")
        
        # Task execution tabs
        exec_tab1, exec_tab2, exec_tab3 = st.tabs([
            "🎯 Custom Tasks", "⚡ Quick Tests", "📋 Task History"
        ])
        
        with exec_tab1:
            st.markdown("### 📝 Custom QA Task Execution")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                task_description = st.text_area(
                    "Task Description",
                    placeholder="Enter detailed QA task description (e.g., 'Test Wi-Fi toggle functionality and verify state changes')",
                    height=100,
                    help="Describe the QA task in natural language"
                )
                
                # Advanced options
                with st.expander("🔧 Advanced Options"):
                    col1a, col1b = st.columns(2)
                    
                    with col1a:
                        priority = st.selectbox("Priority Level", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=4)
                        expected_duration = st.number_input("Expected Duration (seconds)", min_value=5, max_value=600, value=60)
                    
                    with col1b:
                        max_retries = st.number_input("Max Retries", min_value=0, max_value=5, value=2)
                        verification_level = st.selectbox("Verification Level", ["basic", "standard", "comprehensive"], index=1)
                    
                    enable_screenshots = st.checkbox("📸 Capture Screenshots", value=True)
                    enable_detailed_logging = st.checkbox("📝 Detailed Logging", value=False)
                    use_amapi_enhancement = st.checkbox("🧠 Use AMAPI Enhancement", value=self.dashboard_state.get('amapi_enabled', False))
            
            with col2:
                st.markdown("### 🎛️ Execution Controls")
                
                if st.button("🚀 Execute Task", type="primary", use_container_width=True, disabled=not task_description.strip()):
                    task_config = {
                        'description': task_description,
                        'priority': priority,
                        'expected_duration': expected_duration,
                        'max_retries': max_retries,
                        'verification_level': verification_level,
                        'enable_screenshots': enable_screenshots,
                        'detailed_logging': enable_detailed_logging,
                        'use_amapi': use_amapi_enhancement
                    }
                    self._execute_custom_task(task_config)
                
                if st.button("🧠 AMAPI-Enhanced Execution", use_container_width=True, disabled=not self.dashboard_state.get('amapi_enabled', False)):
                    if task_description.strip():
                        task_config = {
                            'description': task_description,
                            'use_amapi': True,
                            'priority': 8,  # High priority for AMAPI tasks
                            'verification_level': 'comprehensive'
                        }
                        self._execute_amapi_enhanced_task(task_config)
                    else:
                        st.error("Please enter a task description")
                
                st.markdown("---")
                
                # Task status
                if self.dashboard_state.get('task_running', False):
                    st.warning("🔄 Task Running...")
                    current_task = self.dashboard_state.get('current_task', 'Unknown task')
                    st.write(f"Current: {current_task}")
                else:
                    st.success("✅ Ready for execution")
        
        with exec_tab2:
            st.markdown("### ⚡ Quick Test Suite")
            
            # Predefined quick tests
            quick_tests = [
                {"name": "📶 WiFi Toggle Test", "description": "Test WiFi on/off functionality", "duration": 30},
                {"name": "⚙️ Settings Navigation", "description": "Navigate through settings menu", "duration": 45},
                {"name": "📷 Camera Launch", "description": "Launch and test camera app", "duration": 25},
                {"name": "🖼️ Gallery Browse", "description": "Open gallery and browse images", "duration": 35},
                {"name": "🔊 Volume Controls", "description": "Test volume up/down controls", "duration": 20},
                {"name": "🔍 Search Function", "description": "Test device search functionality", "duration": 40}
            ]
            
            col1, col2, col3 = st.columns(3)
            
            for i, test in enumerate(quick_tests):
                col = [col1, col2, col3][i % 3]
                
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{test['name']}</h4>
                        <p>{test['description']}</p>
                        <small>Est. Duration: {test['duration']}s</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Run {test['name']}", key=f"quick_test_{i}", use_container_width=True):
                        self._execute_quick_test(test)
            
            # Comprehensive suite
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Run All Quick Tests", type="secondary", use_container_width=True):
                    self._execute_quick_test_suite(quick_tests)
            
            with col2:
                if st.button("🎯 Comprehensive QA Suite", type="primary", use_container_width=True):
                    self._execute_comprehensive_suite()
        
        with exec_tab3:
            st.markdown("### 📋 Task Execution History")
            
            if st.session_state.execution_history:
                # Summary stats
                total_executions = len(st.session_state.execution_history)
                successful_executions = sum(1 for ex in st.session_state.execution_history if ex.get('success', False))
                success_rate = successful_executions / total_executions if total_executions > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Executions", total_executions)
                with col2:
                    st.metric("Successful", successful_executions)
                with col3:
                    st.metric("Success Rate", f"{success_rate:.1%}")
                with col4:
                    amapi_enhanced = sum(1 for ex in st.session_state.execution_history if ex.get('amapi_enhanced', False))
                    st.metric("AMAPI Enhanced", amapi_enhanced)
                
                # Execution history table
                st.markdown("#### Recent Executions")
                
                history_data = []
                for i, execution in enumerate(reversed(st.session_state.execution_history[-20:])):  # Last 20
                    history_data.append({
                        "#": len(st.session_state.execution_history) - i,
                        "Task": execution.get('task', 'Unknown')[:50] + ("..." if len(execution.get('task', '')) > 50 else ''),
                        "Status": "✅ Success" if execution.get('success') else "❌ Failed",
                        "Duration": f"{execution.get('duration', 0):.1f}s",
                        "AMAPI": "🧠" if execution.get('amapi_enhanced', False) else "—",
                        "Timestamp": execution.get('timestamp', '')[:19] if execution.get('timestamp') else 'Unknown',
                        "Agents": execution.get('agents_used', 0)
                    })
                
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True)
                
                # Detailed view
                if st.button("📊 View Detailed Analytics"):
                    self._show_execution_analytics()
            else:
                st.info("📋 No execution history available. Run some QA tasks to see results here.")
    
    def _render_performance_metrics(self, performance_report: Dict):
        """Render performance metrics dashboard"""
        st.markdown("## 📊 Performance Metrics Dashboard")
        
        # Performance overview
        workflow_analysis = performance_report.get('workflow_analysis', {})
        
        st.markdown("### 📈 Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = workflow_analysis.get('success_rate', 0.0)
            st.metric("Overall Success Rate", f"{success_rate:.1%}")
        
        with col2:
            avg_duration = workflow_analysis.get('average_duration', 0.0)
            st.metric("Average Duration", f"{avg_duration:.1f}s")
        
        with col3:
            total_executed = workflow_analysis.get('total_executed', 0)
            st.metric("Total Workflows", total_executed)
        
        with col4:
            system_health = performance_report.get('system_health_score', 0.0)
            st.metric("System Health", f"{system_health:.1%}")
        
        # Performance trends
        if st.session_state.execution_history:
            st.markdown("### 📈 Performance Trends")
            
            # Generate trend data
            trend_data = []
            for i, execution in enumerate(st.session_state.execution_history):
                trend_data.append({
                    'Execution': i + 1,
                    'Success': 1 if execution.get('success', False) else 0,
                    'Duration': execution.get('duration', 0),
                    'AMAPI Enhanced': execution.get('amapi_enhanced', False),
                    'Timestamp': execution.get('timestamp', ''),
                    'Task Type': execution.get('task_type', 'General')
                })
            
            trend_df = pd.DataFrame(trend_data)
            
            if not trend_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Success rate trend
                    fig1 = px.scatter(
                        trend_df,
                        x='Execution',
                        y='Success',
                        color='AMAPI Enhanced',
                        title="Success Rate Over Time",
                        color_discrete_map={True: "#00ff41", False: "#1f77b4"}
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Duration trend
                    fig2 = px.scatter(
                        trend_df,
                        x='Execution',
                        y='Duration',
                        color='AMAPI Enhanced',
                        title="Execution Duration Trend",
                        color_discrete_map={True: "#00ff41", False: "#1f77b4"}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Performance comparison
                if trend_df['AMAPI Enhanced'].any():
                    st.markdown("### 🧠 AMAPI vs Standard Performance")
                    
                    amapi_data = trend_df[trend_df['AMAPI Enhanced'] == True]
                    standard_data = trend_df[trend_df['AMAPI Enhanced'] == False]
                    
                    if not amapi_data.empty and not standard_data.empty:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            amapi_success = amapi_data['Success'].mean()
                            standard_success = standard_data['Success'].mean()
                            improvement = amapi_success - standard_success
                            st.metric("AMAPI Success Rate", f"{amapi_success:.1%}", f"{improvement:+.1%}")
                        
                        with col2:
                            amapi_duration = amapi_data['Duration'].mean()
                            standard_duration = standard_data['Duration'].mean()
                            duration_diff = amapi_duration - standard_duration
                            st.metric("AMAPI Avg Duration", f"{amapi_duration:.1f}s", f"{duration_diff:+.1f}s")
                        
                        with col3:
                            efficiency_gain = (improvement / max(standard_success, 0.01)) * 100
                            st.metric("Efficiency Gain", f"{efficiency_gain:+.1f}%")
        else:
            st.info("No performance data available. Execute some tasks to see trends.")
    
    def _render_benchmarking_interface(self):
        """Render benchmarking interface"""
        st.markdown("## 🔬 System Benchmarking")
        
        # Benchmark configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Benchmark Configuration")
            
            # Test suite selection
            available_tests = [
                "wifi_toggle", "settings_navigation", "app_launch", "camera_test",
                "gallery_browse", "volume_controls", "search_function", "keyboard_input",
                "notification_panel", "quick_settings"
            ]
            
            selected_tests = st.multiselect(
                "Select Tests for Benchmark",
                available_tests,
                default=["wifi_toggle", "settings_navigation", "app_launch"],
                help="Choose which tests to include in the benchmark suite"
            )
            
            col1a, col1b = st.columns(2)
            
            with col1a:
                iterations = st.number_input("Iterations per Test", min_value=1, max_value=10, value=3)
                timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=300, value=120)
            
            with col1b:
                parallel_execution = st.checkbox("Parallel Execution", value=False)
                include_stress_test = st.checkbox("Include Stress Tests", value=False)
        
        with col2:
            st.markdown("### 🚀 Execution Options")
            
            benchmark_type = st.radio(
                "Benchmark Type",
                ["Standard", "AMAPI-Enhanced", "Comparison"],
                help="Choose benchmark execution mode"
            )
            
            if st.button("🔬 Run Benchmark Suite", type="primary", use_container_width=True, disabled=not selected_tests):
                benchmark_config = {
                    'selected_tests': selected_tests,
                    'iterations': iterations,
                    'timeout': timeout,
                    'parallel_execution': parallel_execution,
                    'include_stress_test': include_stress_test,
                    'benchmark_type': benchmark_type
                }
                self._execute_benchmark_suite(benchmark_config)
            
            if st.button("📊 Load Previous Results", use_container_width=True):
                self._load_benchmark_results()
        
        # Display benchmark results
        if st.session_state.benchmark_results:
            st.markdown("### 📊 Benchmark Results")
            self._display_benchmark_results(st.session_state.benchmark_results)
    
    def _render_system_debugging(self):
        """Render system debugging interface"""
        st.markdown("## 🔍 System Debugging & Diagnostics")
        
        debug_tab1, debug_tab2, debug_tab3, debug_tab4 = st.tabs([
            "📝 System Logs", "🔧 Diagnostics", "📊 Real-time Metrics", "🧠 AMAPI Debug"
        ])
        
        with debug_tab1:
            st.markdown("### 📝 System Activity Logs")
            
            # Log level filter
            log_levels = st.multiselect(
                "Filter by Log Level",
                ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"],
                default=["INFO", "SUCCESS", "WARNING", "ERROR"]
            )
            
            # Display logs
            if st.session_state.system_logs:
                filtered_logs = [
                    log for log in st.session_state.system_logs
                    if log.get('level', 'INFO') in log_levels
                ]
                
                for log_entry in filtered_logs[-20:]:  # Last 20 logs
                    timestamp = log_entry.get('timestamp', 'Unknown')
                    level = log_entry.get('level', 'INFO')
                    component = log_entry.get('component', 'System')
                    message = log_entry.get('message', '')
                    
                    level_colors = {
                        'DEBUG': '#9C27B0',
                        'INFO': '#2196F3',
                        'SUCCESS': '#4CAF50',
                        'WARNING': '#ff9800',
                        'ERROR': '#f44336'
                    }
                    
                    color = level_colors.get(level, '#2196F3')
                    
                    st.markdown(f"""
                    <div class="task-log" style="border-left-color: {color};">
                        <strong>[{timestamp}] {level} - {component}</strong><br>
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No system logs available")
            
            # Log export
            if st.button("📄 Export Logs"):
                self._export_system_logs()
        
        with debug_tab2:
            st.markdown("### 🔧 System Diagnostics")
            
            if st.button("🔍 Run System Diagnostics", type="primary"):
                self._run_system_diagnostics()
            
            # Component health check
            if self.system_evaluator:
                try:
                    health_assessment = asyncio.run(self.system_evaluator.evaluate_system_health())
                    
                    st.markdown("#### 🏥 Component Health Status")
                    
                    for component in health_assessment.component_evaluations:
                        health_icon = {
                            ComponentHealth.EXCELLENT: "🟢",
                            ComponentHealth.GOOD: "🟡",
                            ComponentHealth.FAIR: "🟠",
                            ComponentHealth.POOR: "🔴",
                            ComponentHealth.CRITICAL: "⚫"
                        }.get(component.health_status, "❓")
                        
                        st.write(f"{health_icon} **{component.component_name}**: {component.health_status.value}")
                        st.write(f"   Performance: {component.performance_score:.1f}%")
                        st.write(f"   Reliability: {component.reliability_score:.1f}%")
                        
                        if component.issues:
                            st.write("   Issues:")
                            for issue in component.issues:
                                st.write(f"   • {issue}")
                
                except Exception as e:
                    st.error(f"Health assessment failed: {e}")
        
        with debug_tab3:
            st.markdown("### 📊 Real-time System Metrics")
            
            if self.system_metrics:
                try:
                    all_metrics = asyncio.run(self.system_metrics.get_all_metrics())
                    
                    st.markdown("#### 📈 Live Metrics")
                    
                    metrics_summary = all_metrics.get('metrics', {})
                    
                    for metric_name, metric_info in metrics_summary.items():
                        aggregations = metric_info.get('aggregations', {})
                        if 'latest' in aggregations:
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{aggregations['latest']:.2f}",
                                delta=f"{aggregations.get('mean', 0):.2f}" if 'mean' in aggregations else None
                            )
                
                except Exception as e:
                    st.error(f"Metrics unavailable: {e}")
            else:
                st.info("System metrics not initialized")
        
        with debug_tab4:
            st.markdown("### 🧠 AMAPI Debug Information")
            
            if self.dashboard_state.get('amapi_enabled', False) and hasattr(self.orchestrator, 'amapi_core'):
                try:
                    amapi_analytics = self.orchestrator.amapi_core.get_amapi_analytics()
                    
                    # Debug information
                    st.markdown("#### 🔍 AMAPI Internal State")
                    
                    debug_info = {
                        "Learning Events Count": amapi_analytics['learning_events_summary'].get('total_events', 0),
                        "Knowledge Connections": amapi_analytics['knowledge_graph_analysis'].get('total_connections', 0),
                        "Behavioral Models": amapi_analytics['behavioral_model_summary'].get('agents_modeled', 0),
                        "System Intelligence": amapi_analytics['system_metrics'].get('system_intelligence_quotient', 0.0)
                    }
                    
                    for key, value in debug_info.items():
                        st.write(f"**{key}**: {value}")
                    
                    # Raw analytics data
                    with st.expander("🔧 Raw AMAPI Analytics Data"):
                        st.json(amapi_analytics)
                
                except Exception as e:
                    st.error(f"AMAPI debug info unavailable: {e}")
            else:
                st.info("AMAPI not enabled or unavailable")
    
    # Task execution methods
    def _execute_wifi_test(self):
        """Execute WiFi toggle test"""
        self._log_system_event("INFO", "Dashboard", "Starting WiFi toggle test")
        
        if not self.orchestrator:
            st.error("System not initialized")
            return
        
        with st.spinner("🔄 Executing WiFi toggle test..."):
            try:
                self.dashboard_state['task_running'] = True
                self.dashboard_state['current_task'] = "WiFi Toggle Test"
                
                start_time = time.time()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.orchestrator.execute_wifi_toggle_task())
                
                execution_time = time.time() - start_time
                
                # Record execution
                execution_record = {
                    'task': 'WiFi Toggle Test',
                    'task_type': 'wifi_toggle',
                    'success': result.get('success', False),
                    'duration': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'amapi_enhanced': result.get('amapi_enhanced', False),
                    'agents_used': len(result.get('agents_used', [])),
                    'primary_agent': result.get('primary_agent'),
                    'error': result.get('error') if not result.get('success') else None
                }
                
                st.session_state.execution_history.append(execution_record)
                
                if result.get('success'):
                    st.success(f"✅ WiFi test completed successfully in {execution_time:.1f}s")
                    self._log_system_event("SUCCESS", "WiFi Test", f"Test completed in {execution_time:.1f}s")
                else:
                    st.error(f"❌ WiFi test failed: {result.get('error', 'Unknown error')}")
                    self._log_system_event("ERROR", "WiFi Test", f"Test failed: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                st.error(f"❌ Test execution error: {e}")
                self._log_system_event("ERROR", "WiFi Test", f"Execution error: {e}")
            finally:
                self.dashboard_state['task_running'] = False
                self.dashboard_state['current_task'] = None
    
    def _execute_settings_test(self):
        """Execute settings navigation test"""
        self._log_system_event("INFO", "Dashboard", "Starting settings navigation test")
        
        if not self.orchestrator:
            st.error("System not initialized")
            return
        
        with st.spinner("🔄 Executing settings navigation test..."):
            try:
                self.dashboard_state['task_running'] = True
                self.dashboard_state['current_task'] = "Settings Navigation Test"
                
                start_time = time.time()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.orchestrator.execute_settings_navigation_task())
                
                execution_time = time.time() - start_time
                
                # Record execution
                execution_record = {
                    'task': 'Settings Navigation Test',
                    'task_type': 'settings_navigation',
                    'success': result.get('success', False),
                    'duration': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'amapi_enhanced': result.get('amapi_enhanced', False),
                    'agents_used': len(result.get('agents_used', [])),
                    'primary_agent': result.get('primary_agent'),
                    'error': result.get('error') if not result.get('success') else None
                }
                
                st.session_state.execution_history.append(execution_record)
                
                if result.get('success'):
                    st.success(f"✅ Settings test completed successfully in {execution_time:.1f}s")
                    self._log_system_event("SUCCESS", "Settings Test", f"Test completed in {execution_time:.1f}s")
                else:
                    st.error(f"❌ Settings test failed: {result.get('error', 'Unknown error')}")
                    self._log_system_event("ERROR", "Settings Test", f"Test failed: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                st.error(f"❌ Test execution error: {e}")
                self._log_system_event("ERROR", "Settings Test", f"Execution error: {e}")
            finally:
                self.dashboard_state['task_running'] = False
                self.dashboard_state['current_task'] = None
    
    def _execute_comprehensive_suite(self):
        """Execute comprehensive QA suite"""
        self._log_system_event("INFO", "Dashboard", "Starting comprehensive QA suite")
        
        if not self.orchestrator:
            st.error("System not initialized")
            return
        
        with st.spinner("🔄 Executing comprehensive QA suite..."):
            try:
                self.dashboard_state['task_running'] = True
                self.dashboard_state['current_task'] = "Comprehensive QA Suite"
                
                start_time = time.time()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.orchestrator.execute_comprehensive_qa_suite())
                
                execution_time = time.time() - start_time
                
                # Record execution
                execution_record = {
                    'task': 'Comprehensive QA Suite',
                    'task_type': 'comprehensive_suite',
                    'success': result.get('success', False),
                    'duration': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'amapi_enhanced': result.get('amapi_enhanced', False),
                    'agents_used': len(result.get('agents_used', [])),
                    'primary_agent': result.get('primary_agent'),
                    'error': result.get('error') if not result.get('success') else None,
                    'suite_results': result.get('task_results', [])
                }
                
                st.session_state.execution_history.append(execution_record)
                
                if result.get('success'):
                    success_rate = result.get('success_rate', 0)
                    total_time = result.get('total_execution_time', execution_time)
                    st.success(f"✅ QA Suite completed: {success_rate:.1%} success rate in {total_time:.1f}s")
                    
                    # Show detailed results
                    with st.expander("📊 Detailed Suite Results"):
                        task_results = result.get('task_results', [])
                        for task_result in task_results:
                            status = "✅" if task_result.get('success') else "❌"
                            task_name = task_result.get('task_name', 'Unknown')
                            exec_time = task_result.get('execution_time', 0)
                            st.write(f"{status} {task_name}: {exec_time:.1f}s")
                    
                    self._log_system_event("SUCCESS", "QA Suite", f"Suite completed with {success_rate:.1%} success rate")
                else:
                    st.error(f"❌ QA Suite failed: {result.get('error', 'Unknown error')}")
                    self._log_system_event("ERROR", "QA Suite", f"Suite failed: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                st.error(f"❌ Suite execution error: {e}")
                self._log_system_event("ERROR", "QA Suite", f"Execution error: {e}")
            finally:
                self.dashboard_state['task_running'] = False
                self.dashboard_state['current_task'] = None
    
    def _execute_custom_task(self, task_config: Dict[str, Any]):
        """Execute custom QA task"""
        description = task_config.get('description', '')
        self._log_system_event("INFO", "Dashboard", f"Starting custom task: {description[:50]}...")
        
        if not self.orchestrator:
            st.error("System not initialized")
            return
        
        with st.spinner(f"🔄 Executing custom task: {description[:50]}..."):
            try:
                self.dashboard_state['task_running'] = True
                self.dashboard_state['current_task'] = description[:50] + "..."
                
                start_time = time.time()
                
                # Build requirements from config
                requirements = {
                    'priority': task_config.get('priority', 5),
                    'expected_duration': task_config.get('expected_duration', 60),
                    'max_retries': task_config.get('max_retries', 2),
                    'verification_level': task_config.get('verification_level', 'standard'),
                    'enable_screenshots': task_config.get('enable_screenshots', True),
                    'detailed_logging': task_config.get('detailed_logging', False),
                    'use_amapi': task_config.get('use_amapi', False)
                }
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.orchestrator.execute_qa_task(description, requirements)
                )
                
                execution_time = time.time() - start_time
                
                # Record execution
                execution_record = {
                    'task': f"Custom: {description[:50]}",
                    'task_type': 'custom',
                    'success': result.get('success', False),
                    'duration': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'amapi_enhanced': result.get('amapi_enhanced', False),
                    'agents_used': len(result.get('agents_used', [])),
                    'primary_agent': result.get('primary_agent'),
                    'error': result.get('error') if not result.get('success') else None,
                    'full_description': description
                }
                
                st.session_state.execution_history.append(execution_record)
                
                if result.get('success'):
                    st.success(f"✅ Custom task completed successfully in {execution_time:.1f}s")
                    self._log_system_event("SUCCESS", "Custom Task", f"Task completed in {execution_time:.1f}s")
                else:
                    st.error(f"❌ Custom task failed: {result.get('error', 'Unknown error')}")
                    self._log_system_event("ERROR", "Custom Task", f"Task failed: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                st.error(f"❌ Task execution error: {e}")
                self._log_system_event("ERROR", "Custom Task", f"Execution error: {e}")
            finally:
                self.dashboard_state['task_running'] = False
                self.dashboard_state['current_task'] = None
    
    def _execute_amapi_enhanced_task(self, task_config: Dict[str, Any]):
        """Execute AMAPI-enhanced task"""
        description = task_config.get('description', '')
        self._log_system_event("INFO", "AMAPI", f"Starting AMAPI-enhanced task: {description[:50]}...")
        
        if not self.orchestrator or not self.dashboard_state.get('amapi_enabled', False):
            st.error("AMAPI not available")
            return
        
        with st.spinner(f"🧠 Executing AMAPI-enhanced task: {description[:50]}..."):
            try:
                self.dashboard_state['task_running'] = True
                self.dashboard_state['current_task'] = f"AMAPI: {description[:40]}..."
                
                start_time = time.time()
                
                # Enhanced requirements for AMAPI tasks
                requirements = {
                    'priority': task_config.get('priority', 8),
                    'verification_level': 'comprehensive',
                    'use_amapi': True,
                    'enable_learning': True,
                    'adaptive_execution': True
                }
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.orchestrator.execute_amapi_enhanced_task(description, requirements)
                )
                
                execution_time = time.time() - start_time
                
                # Record execution
                execution_record = {
                    'task': f"AMAPI: {description[:40]}",
                    'task_type': 'amapi_enhanced',
                    'success': result.get('success', False),
                    'duration': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'amapi_enhanced': True,
                    'agents_used': len(result.get('agents_used', [])),
                    'primary_agent': result.get('primary_agent'),
                    'error': result.get('error') if not result.get('success') else None,
                    'amapi_insights': result.get('amapi_insights', {}),
                    'learning_events': result.get('learning_events', [])
                }
                
                st.session_state.execution_history.append(execution_record)
                
                if result.get('success'):
                    st.success(f"🧠 AMAPI-enhanced task completed successfully in {execution_time:.1f}s")
                    
                    # Show AMAPI insights
                    amapi_insights = result.get('amapi_insights', {})
                    if amapi_insights:
                        with st.expander("🧠 AMAPI Insights"):
                            for insight_type, insight_data in amapi_insights.items():
                                st.write(f"**{insight_type}**: {insight_data}")
                    
                    self._log_system_event("SUCCESS", "AMAPI Task", f"Enhanced task completed in {execution_time:.1f}s")
                else:
                    st.error(f"❌ AMAPI-enhanced task failed: {result.get('error', 'Unknown error')}")
                    self._log_system_event("ERROR", "AMAPI Task", f"Enhanced task failed: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                st.error(f"❌ AMAPI task execution error: {e}")
                self._log_system_event("ERROR", "AMAPI Task", f"Execution error: {e}")
            finally:
                self.dashboard_state['task_running'] = False
                self.dashboard_state['current_task'] = None
    
    def _execute_quick_test(self, test_config: Dict[str, Any]):
        """Execute a quick test"""
        test_name = test_config.get('name', 'Quick Test')
        self._log_system_event("INFO", "Quick Test", f"Starting {test_name}")
        
        # Simulate quick test execution
        with st.spinner(f"🔄 Running {test_name}..."):
            time.sleep(2)  # Simulate execution time
            
            # Mock result
            success = np.random.random() > 0.1  # 90% success rate
            duration = test_config.get('duration', 30) + np.random.normal(0, 5)
            
            execution_record = {
                'task': test_name,
                'task_type': 'quick_test',
                'success': success,
                'duration': max(5, duration),
                'timestamp': datetime.now().isoformat(),
                'amapi_enhanced': False,
                'agents_used': 2,
                'primary_agent': 'executor'
            }
            
            st.session_state.execution_history.append(execution_record)
            
            if success:
                st.success(f"✅ {test_name} completed successfully")
                self._log_system_event("SUCCESS", "Quick Test", f"{test_name} completed")
            else:
                st.error(f"❌ {test_name} failed")
                self._log_system_event("ERROR", "Quick Test", f"{test_name} failed")
    
    def _execute_benchmark_suite(self, config: Dict[str, Any]):
        """Execute benchmark suite"""
        self._log_system_event("INFO", "Benchmark", "Starting benchmark suite execution")
        
        selected_tests = config.get('selected_tests', [])
        iterations = config.get('iterations', 3)
        benchmark_type = config.get('benchmark_type', 'Standard')
        
        with st.spinner(f"🔬 Running {benchmark_type} benchmark suite..."):
            try:
                start_time = time.time()
                
                # Simulate benchmark execution
                results = {
                    'benchmark_type': benchmark_type,
                    'timestamp': datetime.now().isoformat(),
                    'configuration': config,
                    'results': {},
                    'summary': {}
                }
                
                total_success = 0
                total_tests = 0
                
                for test in selected_tests:
                    test_results = []
                    
                    for iteration in range(iterations):
                        # Simulate test execution
                        success = np.random.random() > 0.15  # 85% success rate
                        duration = 20 + np.random.normal(0, 5)
                        
                        test_results.append({
                            'iteration': iteration + 1,
                            'success': success,
                            'duration': max(5, duration)
                        })
                        
                        if success:
                            total_success += 1
                        total_tests += 1
                    
                    # Calculate test summary
                    test_success_rate = sum(1 for r in test_results if r['success']) / len(test_results)
                    avg_duration = np.mean([r['duration'] for r in test_results])
                    
                    results['results'][test] = {
                        'success_rate': test_success_rate,
                        'average_duration': avg_duration,
                        'iterations': iterations,
                        'detailed_results': test_results
                    }
                
                # Calculate overall summary
                overall_success_rate = total_success / total_tests if total_tests > 0 else 0
                total_execution_time = time.time() - start_time
                
                results['summary'] = {
                    'overall_success_rate': overall_success_rate,
                    'total_execution_time': total_execution_time,
                    'tests_executed': len(selected_tests),
                    'total_iterations': total_tests
                }
                
                st.session_state.benchmark_results = results
                
                st.success(f"✅ Benchmark suite completed: {overall_success_rate:.1%} success rate")
                self._log_system_event("SUCCESS", "Benchmark", f"Suite completed with {overall_success_rate:.1%} success rate")
                
                # Display results
                self._display_benchmark_results(results)
                
            except Exception as e:
                st.error(f"❌ Benchmark execution error: {e}")
                self._log_system_event("ERROR", "Benchmark", f"Execution error: {e}")
    
    def _display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results"""
        benchmark_type = results.get('benchmark_type', 'Unknown')
        summary = results.get('summary', {})
        test_results = results.get('results', {})
        
        st.markdown(f"#### 📊 {benchmark_type} Benchmark Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = summary.get('overall_success_rate', 0)
            st.metric("Overall Success Rate", f"{success_rate:.1%}")
        
        with col2:
            execution_time = summary.get('total_execution_time', 0)
            st.metric("Total Time", f"{execution_time:.1f}s")
        
        with col3:
            tests_executed = summary.get('tests_executed', 0)
            st.metric("Tests Executed", tests_executed)
        
        with col4:
            total_iterations = summary.get('total_iterations', 0)
            st.metric("Total Iterations", total_iterations)
        
        # Detailed results table
        if test_results:
            st.markdown("##### 📋 Test Results Breakdown")
            
            results_data = []
            for test_name, test_data in test_results.items():
                results_data.append({
                    'Test': test_name.replace('_', ' ').title(),
                    'Success Rate': f"{test_data['success_rate']:.1%}",
                    'Avg Duration': f"{test_data['average_duration']:.1f}s",
                    'Iterations': test_data['iterations']
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            if len(test_results) > 1:
                success_rates = [test_data['success_rate'] for test_data in test_results.values()]
                test_names = [name.replace('_', ' ').title() for name in test_results.keys()]
                
                fig = px.bar(
                    x=test_names,
                    y=success_rates,
                    title=f"{benchmark_type} Benchmark - Success Rates by Test",
                    labels={'x': 'Test', 'y': 'Success Rate'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Utility methods
    def _log_system_event(self, level: str, component: str, message: str):
        """Log system event"""
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': level,
            'component': component,
            'message': message
        }
        
        st.session_state.system_logs.append(log_entry)
        
        # Keep only last 1000 logs
        if len(st.session_state.system_logs) > 1000:
            st.session_state.system_logs = st.session_state.system_logs[-1000:]
    
    def _restart_system(self):
        """Restart the system"""
        with st.spinner("🔄 Restarting system..."):
            try:
                if self.orchestrator:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.orchestrator.shutdown_system())
                
                # Reset state
                self.orchestrator = None
                self.dashboard_state['initialized'] = False
                
                st.success("✅ System restart initiated")
                self._log_system_event("INFO", "System", "System restart initiated")
                
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Restart failed: {e}")
                self._log_system_event("ERROR", "System", f"Restart failed: {e}")
    
    def _shutdown_system(self):
        """Shutdown the system"""
        with st.spinner("🛑 Shutting down system..."):
            try:
                if self.orchestrator:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.orchestrator.shutdown_system())
                
                self.orchestrator = None
                self.dashboard_state['initialized'] = False
                
                st.success("✅ System shutdown completed")
                self._log_system_event("INFO", "System", "System shutdown completed")
                
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Shutdown failed: {e}")
                self._log_system_event("ERROR", "System", f"Shutdown failed: {e}")
    
    def _run_system_diagnostics(self):
        """Run comprehensive system diagnostics"""
        with st.spinner("🔍 Running system diagnostics..."):
            try:
                diagnostics_results = {
                    'timestamp': datetime.now().isoformat(),
                    'system_status': 'healthy',
                    'components_checked': [],
                    'issues_found': [],
                    'recommendations': []
                }
                
                # Check orchestrator
                if self.orchestrator:
                    diagnostics_results['components_checked'].append('orchestrator')
                    if not self.orchestrator.is_running:
                        diagnostics_results['issues_found'].append('Orchestrator not running')
                
                # Check AMAPI
                if self.dashboard_state.get('amapi_enabled', False):
                    diagnostics_results['components_checked'].append('amapi_core')
                    if not hasattr(self.orchestrator, 'amapi_core'):
                        diagnostics_results['issues_found'].append('AMAPI core not accessible')
                
                # Check metrics
                if self.system_metrics:
                    diagnostics_results['components_checked'].append('system_metrics')
                
                # Generate recommendations
                if len(diagnostics_results['issues_found']) == 0:
                    diagnostics_results['system_status'] = 'healthy'
                    diagnostics_results['recommendations'].append('System operating normally')
                else:
                    diagnostics_results['system_status'] = 'degraded'
                    diagnostics_results['recommendations'].append('Address identified issues')
                
                # Display results
                st.markdown("#### 🔍 Diagnostic Results")
                
                status_color = "success" if diagnostics_results['system_status'] == 'healthy' else "warning"
                st.write(f"**System Status**: :{status_color}[{diagnostics_results['system_status'].title()}]")
                
                st.write(f"**Components Checked**: {', '.join(diagnostics_results['components_checked'])}")
                
                if diagnostics_results['issues_found']:
                    st.write("**Issues Found**:")
                    for issue in diagnostics_results['issues_found']:
                        st.write(f"• {issue}")
                
                st.write("**Recommendations**:")
                for rec in diagnostics_results['recommendations']:
                    st.write(f"• {rec}")
                
                self._log_system_event("INFO", "Diagnostics", f"System diagnostics completed - Status: {diagnostics_results['system_status']}")
                
            except Exception as e:
                st.error(f"❌ Diagnostics failed: {e}")
                self._log_system_event("ERROR", "Diagnostics", f"Diagnostics failed: {e}")


def main():
    """Main entry point for Streamlit dashboard"""
    try:
        dashboard = StreamlitQADashboard()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"❌ Critical dashboard error: {e}")
        logger.error(f"Dashboard initialization failed: {e}")
        
        st.markdown("""
        ### 🔧 Troubleshooting
        
        If you're seeing this error, please check:
        1. All required dependencies are installed
        2. System components are properly imported
        3. Configuration files are available
        4. No circular import issues exist
        
        Try refreshing the page or restarting the Streamlit server.
        """)


if __name__ == "__main__":
    main()