"""
Production-Ready Verifier Agent for Multi-Agent QA System
Verifies execution results and detects bugs with AMAPI learning
"""

import time
import json
import uuid
import asyncio
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from loguru import logger
import numpy as np

from .base_agent import BaseQAAgent, AgentAction, ActionType


class VerificationResult(Enum):
    """Verification result states"""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class BugSeverity(Enum):
    """Bug severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    COSMETIC = "cosmetic"


class BugType(Enum):
    """Types of bugs that can be detected"""
    FUNCTIONAL = "functional"
    UI = "ui"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    COMPATIBILITY = "compatibility"
    SECURITY = "security"


@dataclass
class BugReport:
    """Detailed bug report"""
    bug_id: str
    bug_type: BugType
    severity: BugSeverity
    title: str
    description: str
    steps_to_reproduce: List[str]
    expected_behavior: str
    actual_behavior: str
    screenshot_path: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class VerificationReport:
    """Comprehensive verification report"""
    report_id: str
    verification_result: VerificationResult
    confidence_score: float
    execution_step_id: str
    subgoal_id: str
    verification_criteria: List[str]
    bugs_detected: List[BugReport]
    performance_metrics: Dict[str, Any]
    verification_time: float
    screenshot_evidence: List[str]
    learned_patterns: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class VerifierAgent(BaseQAAgent):
    """
    Production-ready Verifier Agent with comprehensive bug detection
    Uses multiple verification strategies with AMAPI learning
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("VerifierAgent", "verifier", config)
        
        # Verification strategies
        self.verification_strategies = {
            'ui_hierarchy': self._verify_ui_hierarchy,
            'visual_comparison': self._verify_visual_state,
            'functional_state': self._verify_functional_state,
            'performance_check': self._verify_performance,
            'accessibility': self._verify_accessibility,
            'device_state': self._verify_device_state
        }
        
        # Bug detection patterns
        self.bug_patterns = {
            'crash_patterns': [
                'unfortunately', 'has stopped', 'force close', 'not responding',
                'application error', 'runtime error', 'null pointer'
            ],
            'ui_issues': [
                'overlapping', 'cut off', 'missing button', 'invisible text',
                'wrong color', 'misaligned', 'truncated'
            ],
            'performance_issues': [
                'slow response', 'lag', 'freeze', 'timeout', 'memory leak',
                'high cpu', 'battery drain'
            ]
        }
        
        # Learning data
        self.verification_patterns: Dict[str, Dict[str, Any]] = {}
        self.bug_detection_history: List[BugReport] = []
        self.false_positive_patterns: Dict[str, float] = {}
        self.verification_accuracy_tracking: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.verification_metrics = {
            'total_verifications': 0,
            'passed_verifications': 0,
            'failed_verifications': 0,
            'bugs_detected': 0,
            'false_positives': 0,
            'average_verification_time': 0.0,
            'accuracy_score': 0.0,
            'detection_precision': 0.0
        }
        
        logger.info("VerifierAgent initialized with comprehensive bug detection")

    async def predict(self, instruction: str, observation: Dict[str, Any], 
                     context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Verify execution results and detect bugs
        """
        logger.info(f"VerifierAgent verifying: {instruction}")
        
        try:
            # Parse verification request
            verification_request = await self._parse_verification_request(instruction, context)
            
            # Perform comprehensive verification
            verification_report = await self._perform_comprehensive_verification(
                verification_request, observation, context
            )
            
            # Generate verification actions/recommendations
            actions = await self._generate_verification_actions(verification_report)
            
            reasoning_info = {
                "reasoning": f"Verification completed: {verification_report.verification_result.value}",
                "confidence": verification_report.confidence_score,
                "attention_cost": self._calculate_attention_cost(verification_report),
                "verification_result": verification_report.verification_result.value,
                "bugs_detected": len(verification_report.bugs_detected),
                "report_id": verification_report.report_id,
                "verification_strategies_used": len(self.verification_strategies)
            }
            
            return reasoning_info, actions
            
        except Exception as e:
            logger.error(f"Error in VerifierAgent prediction: {e}")
            return {
                "reasoning": f"Verification error: {str(e)}",
                "confidence": 0.1,
                "error": str(e),
                "verification_result": "error"
            }, ["# Verification failed"]

    async def _parse_verification_request(self, instruction: str, 
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse verification instruction into structured request"""
        instruction_lower = instruction.lower()
        
        request = {
            "verification_type": "general",
            "criteria": [],
            "expected_outcomes": [],
            "bug_detection_focus": [],
            "priority": "medium"
        }
        
        # Determine verification type
        if 'wifi' in instruction_lower or 'network' in instruction_lower:
            request["verification_type"] = "network_functionality"
            request["criteria"] = ["wifi_state_changed", "network_connectivity", "ui_response"]
            request["expected_outcomes"] = ["wifi_toggle_works", "state_indicator_updated"]
            
        elif 'ui' in instruction_lower or 'interface' in instruction_lower:
            request["verification_type"] = "ui_validation"
            request["criteria"] = ["ui_elements_present", "layout_correct", "responsiveness"]
            request["bug_detection_focus"] = ["ui_issues"]
            
        elif 'performance' in instruction_lower:
            request["verification_type"] = "performance_check"
            request["criteria"] = ["response_time", "memory_usage", "cpu_usage"]
            request["bug_detection_focus"] = ["performance_issues"]
            
        elif 'crash' in instruction_lower or 'error' in instruction_lower:
            request["verification_type"] = "stability_check"
            request["criteria"] = ["app_running", "no_crashes", "error_handling"]
            request["bug_detection_focus"] = ["crash_patterns"]
            
        else:
            # General verification
            request["criteria"] = ["basic_functionality", "ui_state", "no_errors"]
            request["expected_outcomes"] = ["task_completed_successfully"]
        
        # Extract specific criteria from context
        if context:
            if 'verification_criteria' in context:
                request["criteria"].extend(context['verification_criteria'])
            if 'expected_result' in context:
                request["expected_outcomes"].append(context['expected_result'])
            if 'subgoal_id' in context:
                request["subgoal_id"] = context['subgoal_id']
            if 'execution_step_id' in context:
                request["execution_step_id"] = context['execution_step_id']
        
        return request

    async def _perform_comprehensive_verification(self, verification_request: Dict[str, Any],
                                                observation: Dict[str, Any],
                                                context: Dict[str, Any] = None) -> VerificationReport:
        """
        Perform comprehensive verification using multiple strategies
        """
        start_time = time.time()
        report_id = f"verification_{uuid.uuid4().hex[:8]}"
        
        # Initialize report
        verification_report = VerificationReport(
            report_id=report_id,
            verification_result=VerificationResult.INCONCLUSIVE,
            confidence_score=0.0,
            execution_step_id=verification_request.get('execution_step_id', 'unknown'),
            subgoal_id=verification_request.get('subgoal_id', 'unknown'),
            verification_criteria=verification_request.get('criteria', []),
            bugs_detected=[],
            performance_metrics={},
            verification_time=0.0,
            screenshot_evidence=[],
            learned_patterns={}
        )
        
        try:
            # Take screenshot for evidence
            screenshot_path = await self._take_verification_screenshot()
            if screenshot_path:
                verification_report.screenshot_evidence.append(screenshot_path)
            
            # Apply verification strategies
            strategy_results = {}
            
            for strategy_name, strategy_func in self.verification_strategies.items():
                try:
                    if self._should_apply_strategy(strategy_name, verification_request):
                        result = await strategy_func(verification_request, observation, context)
                        strategy_results[strategy_name] = result
                except Exception as e:
                    logger.warning(f"Strategy {strategy_name} failed: {e}")
                    strategy_results[strategy_name] = {
                        'success': False,
                        'error': str(e),
                        'confidence': 0.0
                    }
            
            # Aggregate results
            verification_report = await self._aggregate_verification_results(
                verification_report, strategy_results, verification_request
            )
            
            # Detect bugs
            bugs = await self._detect_bugs(verification_request, observation, strategy_results)
            verification_report.bugs_detected = bugs
            
            # Learn from verification
            await self._learn_from_verification(verification_report, strategy_results)
            
            # Update metrics
            verification_report.verification_time = time.time() - start_time
            self._update_verification_metrics(verification_report)
            
            logger.info(f"Verification completed: {verification_report.verification_result.value} "
                       f"({len(bugs)} bugs detected)")
            
            return verification_report
            
        except Exception as e:
            logger.error(f"Comprehensive verification failed: {e}")
            verification_report.verification_result = VerificationResult.ERROR
            verification_report.verification_time = time.time() - start_time
            return verification_report

    def _should_apply_strategy(self, strategy_name: str, verification_request: Dict[str, Any]) -> bool:
        """Determine if a verification strategy should be applied"""
        verification_type = verification_request.get('verification_type', 'general')
        
        # Strategy applicability rules
        if strategy_name == 'ui_hierarchy' and 'ui' in verification_type:
            return True
        elif strategy_name == 'performance_check' and 'performance' in verification_type:
            return True
        elif strategy_name == 'functional_state' and 'network' in verification_type:
            return True
        elif strategy_name == 'visual_comparison':
            return True  # Always useful
        elif strategy_name == 'device_state':
            return True  # Always check device state
        elif strategy_name == 'accessibility' and verification_request.get('priority') == 'high':
            return True
        else:
            return verification_type == 'general'  # Apply to general verifications

    async def _verify_ui_hierarchy(self, verification_request: Dict[str, Any],
                                  observation: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify UI hierarchy and element presence"""
        try:
            # Get UI hierarchy via ADB
            ui_result = await self._run_adb_command([
                "shell", "uiautomator", "dump", "/sdcard/ui_hierarchy.xml"
            ])
            
            if not ui_result['success']:
                return {'success': False, 'confidence': 0.0, 'error': 'UI dump failed'}
            
            # Pull UI hierarchy file
            pull_result = await self._run_adb_command([
                "pull", "/sdcard/ui_hierarchy.xml", "ui_hierarchy.xml"
            ])
            
            if not pull_result['success']:
                return {'success': False, 'confidence': 0.0, 'error': 'UI file pull failed'}
            
            # Analyze UI hierarchy
            ui_analysis = await self._analyze_ui_hierarchy("ui_hierarchy.xml", verification_request)
            
            return {
                'success': True,
                'confidence': ui_analysis.get('confidence', 0.7),
                'ui_elements_found': ui_analysis.get('elements_found', 0),
                'missing_elements': ui_analysis.get('missing_elements', []),
                'ui_issues': ui_analysis.get('issues', [])
            }
            
        except Exception as e:
            logger.error(f"UI hierarchy verification failed: {e}")
            return {'success': False, 'confidence': 0.0, 'error': str(e)}

    async def _verify_visual_state(self, verification_request: Dict[str, Any],
                                  observation: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify visual state through screenshot analysis"""
        try:
            # Take current screenshot
            screenshot_path = await self._take_verification_screenshot()
            
            if not screenshot_path:
                return {'success': False, 'confidence': 0.0, 'error': 'Screenshot failed'}
            
            # Analyze screenshot
            visual_analysis = await self._analyze_screenshot(screenshot_path, verification_request)
            
            return {
                'success': True,
                'confidence': visual_analysis.get('confidence', 0.6),
                'visual_elements_detected': visual_analysis.get('elements', []),
                'visual_issues': visual_analysis.get('issues', []),
                'color_analysis': visual_analysis.get('colors', {}),
                'text_recognition': visual_analysis.get('text', [])
            }
            
        except Exception as e:
            logger.error(f"Visual state verification failed: {e}")
            return {'success': False, 'confidence': 0.0, 'error': str(e)}

    async def _verify_functional_state(self, verification_request: Dict[str, Any],
                                      observation: Dict[str, Any],
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify functional state (e.g., WiFi, system settings)"""
        try:
            verification_type = verification_request.get('verification_type', '')
            functional_checks = {}
            
            # WiFi/Network verification
            if 'network' in verification_type or 'wifi' in verification_type:
                wifi_state = await self._check_wifi_state()
                network_connectivity = await self._check_network_connectivity()
                
                functional_checks.update({
                    'wifi_state': wifi_state,
                    'network_connectivity': network_connectivity,
                    'wifi_functional': wifi_state is not None and network_connectivity
                })
            
            # System state checks
            battery_info = await self._check_battery_status()
            screen_state = await self._check_screen_state()
            
            functional_checks.update({
                'battery_status': battery_info,
                'screen_active': screen_state,
                'system_responsive': True  # If we got this far, system is responsive
            })
            
            # Calculate confidence based on successful checks
            successful_checks = sum(1 for check in functional_checks.values() 
                                  if check is not None and check != False)
            confidence = successful_checks / len(functional_checks) if functional_checks else 0.5
            
            return {
                'success': True,
                'confidence': confidence,
                'functional_checks': functional_checks,
                'system_state': 'healthy' if confidence > 0.7 else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Functional state verification failed: {e}")
            return {'success': False, 'confidence': 0.0, 'error': str(e)}

    async def _verify_performance(self, verification_request: Dict[str, Any],
                                 observation: Dict[str, Any],
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify performance characteristics"""
        try:
            performance_metrics = {}
            
            # Memory usage
            memory_result = await self._run_adb_command([
                "shell", "dumpsys", "meminfo"
            ])
            
            if memory_result['success']:
                memory_analysis = self._parse_memory_info(memory_result['output'])
                performance_metrics['memory'] = memory_analysis
            
            # CPU usage
            cpu_result = await self._run_adb_command([
                "shell", "top", "-n", "1"
            ])
            
            if cpu_result['success']:
                cpu_analysis = self._parse_cpu_info(cpu_result['output'])
                performance_metrics['cpu'] = cpu_analysis
            
            # Response time measurement
            response_time = await self._measure_response_time()
            performance_metrics['response_time'] = response_time
            
            # Evaluate performance
            performance_score = self._evaluate_performance(performance_metrics)
            
            return {
                'success': True,
                'confidence': 0.8,
                'performance_metrics': performance_metrics,
                'performance_score': performance_score,
                'performance_issues': self._identify_performance_issues(performance_metrics)
            }
            
        except Exception as e:
            logger.error(f"Performance verification failed: {e}")
            return {'success': False, 'confidence': 0.0, 'error': str(e)}

    async def _verify_accessibility(self, verification_request: Dict[str, Any],
                                   observation: Dict[str, Any],
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify accessibility features"""
        try:
            accessibility_checks = {}
            
            # Check for accessibility service
            a11y_result = await self._run_adb_command([
                "shell", "dumpsys", "accessibility"
            ])
            
            if a11y_result['success']:
                accessibility_checks['service_active'] = 'enabled' in a11y_result['output'].lower()
            
            # Check UI elements for accessibility attributes
            ui_accessibility = await self._check_ui_accessibility()
            accessibility_checks.update(ui_accessibility)
            
            # Text contrast and readability
            contrast_check = await self._check_text_contrast()
            accessibility_checks['contrast_adequate'] = contrast_check
            
            confidence = sum(1 for check in accessibility_checks.values() if check) / len(accessibility_checks)
            
            return {
                'success': True,
                'confidence': confidence,
                'accessibility_checks': accessibility_checks,
                'accessibility_score': confidence
            }
            
        except Exception as e:
            logger.error(f"Accessibility verification failed: {e}")
            return {'success': False, 'confidence': 0.0, 'error': str(e)}

    async def _verify_device_state(self, verification_request: Dict[str, Any],
                                  observation: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify overall device state"""
        try:
            device_checks = {}
            
            # Device responsiveness
            ping_result = await self._run_adb_command(["shell", "echo", "responsive"])
            device_checks['responsive'] = ping_result['success']
            
            # Storage space
            storage_result = await self._run_adb_command(["shell", "df", "/data"])
            if storage_result['success']:
                storage_info = self._parse_storage_info(storage_result['output'])
                device_checks['storage_adequate'] = storage_info.get('free_percentage', 0) > 10
            
            # System services
            services_result = await self._run_adb_command(["shell", "service", "list"])
            device_checks['services_running'] = services_result['success']
            
            # Overall device health
            health_score = sum(1 for check in device_checks.values() if check) / len(device_checks)
            
            return {
                'success': True,
                'confidence': 0.9,
                'device_checks': device_checks,
                'device_health_score': health_score,
                'device_state': 'healthy' if health_score > 0.8 else 'issues_detected'
            }
            
        except Exception as e:
            logger.error(f"Device state verification failed: {e}")
            return {'success': False, 'confidence': 0.0, 'error': str(e)}

    async def _aggregate_verification_results(self, verification_report: VerificationReport,
                                            strategy_results: Dict[str, Dict[str, Any]],
                                            verification_request: Dict[str, Any]) -> VerificationReport:
        """Aggregate results from multiple verification strategies"""
        
        # Calculate weighted confidence score
        total_weight = 0
        weighted_confidence = 0
        
        strategy_weights = {
            'ui_hierarchy': 0.2,
            'visual_comparison': 0.25,
            'functional_state': 0.3,
            'performance_check': 0.1,
            'accessibility': 0.05,
            'device_state': 0.1
        }
        
        for strategy_name, result in strategy_results.items():
            if result.get('success', False):
                weight = strategy_weights.get(strategy_name, 0.1)
                confidence = result.get('confidence', 0.0)
                weighted_confidence += confidence * weight
                total_weight += weight
        
        verification_report.confidence_score = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Determine overall verification result
        if verification_report.confidence_score >= 0.8:
            verification_report.verification_result = VerificationResult.PASS
        elif verification_report.confidence_score >= 0.6:
            verification_report.verification_result = VerificationResult.PARTIAL
        elif verification_report.confidence_score >= 0.3:
            verification_report.verification_result = VerificationResult.FAIL
        else:
            verification_report.verification_result = VerificationResult.INCONCLUSIVE
        
        # Aggregate performance metrics
        verification_report.performance_metrics = self._aggregate_performance_metrics(strategy_results)
        
        # Store learned patterns
        verification_report.learned_patterns = {
            'strategy_effectiveness': {name: result.get('confidence', 0.0) 
                                     for name, result in strategy_results.items()},
            'verification_patterns': self._extract_verification_patterns(strategy_results),
            'success_indicators': self._identify_success_indicators(strategy_results)
        }
        
        return verification_report

    async def _detect_bugs(self, verification_request: Dict[str, Any],
                          observation: Dict[str, Any],
                          strategy_results: Dict[str, Dict[str, Any]]) -> List[BugReport]:
        """Detect bugs based on verification results"""
        bugs_detected = []
        
        try:
            # Check for crash patterns
            crash_bugs = await self._detect_crash_bugs(strategy_results)
            bugs_detected.extend(crash_bugs)
            
            # Check for UI issues
            ui_bugs = await self._detect_ui_bugs(strategy_results)
            bugs_detected.extend(ui_bugs)
            
            # Check for performance issues
            performance_bugs = await self._detect_performance_bugs(strategy_results)
            bugs_detected.extend(performance_bugs)
            
            # Check for functional bugs
            functional_bugs = await self._detect_functional_bugs(verification_request, strategy_results)
            bugs_detected.extend(functional_bugs)
            
            # Apply learned pattern filtering
            filtered_bugs = await self._filter_bugs_with_learning(bugs_detected)
            
            logger.info(f"Bug detection completed: {len(filtered_bugs)} bugs found")
            return filtered_bugs
            
        except Exception as e:
            logger.error(f"Bug detection failed: {e}")
            return []

    async def _detect_crash_bugs(self, strategy_results: Dict[str, Dict[str, Any]]) -> List[BugReport]:
        """Detect crash-related bugs"""
        bugs = []
        
        # Check logcat for crash patterns
        try:
            logcat_result = await self._run_adb_command([
                "logcat", "-d", "-v", "brief", "*:E"
            ])
            
            if logcat_result['success']:
                logcat_output = logcat_result['output'].lower()
                
                for pattern in self.bug_patterns['crash_patterns']:
                    if pattern in logcat_output:
                        bug = BugReport(
                            bug_id=f"crash_{uuid.uuid4().hex[:8]}",
                            bug_type=BugType.FUNCTIONAL,
                            severity=BugSeverity.CRITICAL,
                            title=f"Application crash detected: {pattern}",
                            description=f"Crash pattern '{pattern}' found in system logs",
                            steps_to_reproduce=["Execute current test case"],
                            expected_behavior="Application should remain stable",
                            actual_behavior=f"Application crashed with pattern: {pattern}",
                            device_info=await self._get_device_info()
                        )
                        bugs.append(bug)
        
        except Exception as e:
            logger.warning(f"Crash detection failed: {e}")
        
        return bugs

    async def _detect_ui_bugs(self, strategy_results: Dict[str, Dict[str, Any]]) -> List[BugReport]:
        """Detect UI-related bugs"""
        bugs = []
        
        # Check UI hierarchy results
        ui_result = strategy_results.get('ui_hierarchy', {})
        if 'ui_issues' in ui_result:
            for issue in ui_result['ui_issues']:
                bug = BugReport(
                    bug_id=f"ui_{uuid.uuid4().hex[:8]}",
                    bug_type=BugType.UI,
                    severity=BugSeverity.MEDIUM,
                    title=f"UI Issue: {issue.get('type', 'Unknown')}",
                    description=issue.get('description', 'UI element issue detected'),
                    steps_to_reproduce=["Navigate to current screen"],
                    expected_behavior="UI elements should be properly displayed",
                    actual_behavior=issue.get('description', 'UI rendering issue'),
                    device_info=await self._get_device_info()
                )
                bugs.append(bug)
        
        # Check visual analysis results
        visual_result = strategy_results.get('visual_comparison', {})
        if 'visual_issues' in visual_result:
            for issue in visual_result['visual_issues']:
                bug = BugReport(
                    bug_id=f"visual_{uuid.uuid4().hex[:8]}",
                    bug_type=BugType.UI,
                    severity=BugSeverity.LOW,
                    title=f"Visual Issue: {issue}",
                    description=f"Visual problem detected: {issue}",
                    steps_to_reproduce=["View current screen"],
                    expected_behavior="Proper visual presentation",
                    actual_behavior=f"Visual issue: {issue}",
                    device_info=await self._get_device_info()
                )
                bugs.append(bug)
        
        return bugs

    async def _detect_performance_bugs(self, strategy_results: Dict[str, Dict[str, Any]]) -> List[BugReport]:
        """Detect performance-related bugs"""
        bugs = []
        
        performance_result = strategy_results.get('performance_check', {})
        if 'performance_issues' in performance_result:
            for issue in performance_result['performance_issues']:
                severity = BugSeverity.HIGH if 'critical' in issue.lower() else BugSeverity.MEDIUM
                
                bug = BugReport(
                    bug_id=f"perf_{uuid.uuid4().hex[:8]}",
                    bug_type=BugType.PERFORMANCE,
                    severity=severity,
                    title=f"Performance Issue: {issue}",
                    description=f"Performance problem detected: {issue}",
                    steps_to_reproduce=["Execute performance-intensive operations"],
                    expected_behavior="Acceptable performance levels",
                    actual_behavior=f"Performance issue: {issue}",
                    device_info=await self._get_device_info()
                )
                bugs.append(bug)
        
        return bugs

    async def _detect_functional_bugs(self, verification_request: Dict[str, Any],
                                    strategy_results: Dict[str, Dict[str, Any]]) -> List[BugReport]:
        """Detect functional bugs based on verification criteria"""
        bugs = []
        
        functional_result = strategy_results.get('functional_state', {})
        if 'functional_checks' in functional_result:
            checks = functional_result['functional_checks']
            
            # Check for failed functional requirements
            for check_name, check_result in checks.items():
                if check_result is False or check_result is None:
                    bug = BugReport(
                        bug_id=f"func_{uuid.uuid4().hex[:8]}",
                        bug_type=BugType.FUNCTIONAL,
                        severity=BugSeverity.HIGH,
                        title=f"Functional Issue: {check_name} failed",
                        description=f"Required functionality '{check_name}' is not working correctly",
                        steps_to_reproduce=verification_request.get('criteria', []),
                        expected_behavior=f"{check_name} should work correctly",
                        actual_behavior=f"{check_name} failed or returned unexpected result",
                        device_info=await self._get_device_info()
                    )
                    bugs.append(bug)
        
        return bugs

    async def _filter_bugs_with_learning(self, bugs: List[BugReport]) -> List[BugReport]:
        """Filter bugs using learned patterns to reduce false positives"""
        filtered_bugs = []
        
        for bug in bugs:
            # Check against false positive patterns
            false_positive_score = 0.0
            
            for pattern, score in self.false_positive_patterns.items():
                if pattern.lower() in bug.description.lower():
                    false_positive_score = max(false_positive_score, score)
            
            # Only include bugs with low false positive probability
            if false_positive_score < 0.7:
                filtered_bugs.append(bug)
            else:
                logger.debug(f"Filtered out potential false positive: {bug.title}")
        
        return filtered_bugs

    async def _learn_from_verification(self, verification_report: VerificationReport,
                                     strategy_results: Dict[str, Dict[str, Any]]) -> None:
        """Learn from verification results to improve future verifications"""
        
        # Learn strategy effectiveness
        for strategy_name, result in strategy_results.items():
            if strategy_name not in self.verification_patterns:
                self.verification_patterns[strategy_name] = {
                    'total_uses': 0,
                    'successful_uses': 0,
                    'average_confidence': 0.0,
                    'effectiveness_score': 0.0
                }
            
            pattern = self.verification_patterns[strategy_name]
            pattern['total_uses'] += 1
            
            if result.get('success', False):
                pattern['successful_uses'] += 1
                confidence = result.get('confidence', 0.0)
                current_avg = pattern['average_confidence']
                pattern['average_confidence'] = (
                    (current_avg * (pattern['total_uses'] - 1) + confidence) / pattern['total_uses']
                )
            
            pattern['effectiveness_score'] = pattern['successful_uses'] / pattern['total_uses']
        
        # Learn from bug detection accuracy
        self.bug_detection_history.extend(verification_report.bugs_detected)
        
        # Store verification accuracy for future learning
        accuracy_record = {
            'timestamp': verification_report.timestamp,
            'confidence_predicted': verification_report.confidence_score,
            'result': verification_report.verification_result.value,
            'bugs_detected': len(verification_report.bugs_detected),
            'strategy_results': {k: v.get('confidence', 0.0) for k, v in strategy_results.items()}
        }
        
        self.verification_accuracy_tracking.append(accuracy_record)
        
        # Keep only recent accuracy records
        if len(self.verification_accuracy_tracking) > 100:
            self.verification_accuracy_tracking = self.verification_accuracy_tracking[-100:]

    # Helper methods
    async def _run_adb_command(self, command: List[str]) -> Dict[str, Any]:
        """Run ADB command safely with timeout"""
        try:
            full_command = ["adb"] + command
            
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
                return {
                    'success': process.returncode == 0,
                    'output': stdout.decode('utf-8', errors='ignore'),
                    'error': stderr.decode('utf-8', errors='ignore') if stderr else None
                }
            except asyncio.TimeoutError:
                process.kill()
                return {'success': False, 'error': 'Command timeout'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _take_verification_screenshot(self) -> Optional[str]:
        """Take screenshot for verification evidence"""
        try:
            timestamp = int(time.time())
            screenshot_name = f"verification_screenshot_{timestamp}.png"
            device_path = f"/sdcard/{screenshot_name}"
            local_path = f"screenshots/{screenshot_name}"
            
            # Ensure screenshot directory exists
            import os
            os.makedirs("screenshots", exist_ok=True)
            
            # Take screenshot on device
            screenshot_result = await self._run_adb_command([
                "shell", "screencap", device_path
            ])
            
            if not screenshot_result['success']:
                return None
            
            # Pull screenshot to local system
            pull_result = await self._run_adb_command([
                "pull", device_path, local_path
            ])
            
            if pull_result['success']:
                # Clean up device screenshot
                await self._run_adb_command(["shell", "rm", device_path])
                return local_path
            
            return None
            
        except Exception as e:
            logger.error(f"Verification screenshot failed: {e}")
            return None

    async def _check_wifi_state(self) -> Optional[str]:
        """Check current WiFi state"""
        try:
            result = await self._run_adb_command([
                "shell", "settings", "get", "global", "wifi_on"
            ])
            
            if result['success']:
                return result['output'].strip()
                
        except Exception as e:
            logger.debug(f"WiFi state check failed: {e}")
        
        return None

    async def _check_network_connectivity(self) -> bool:
        """Check network connectivity"""
        try:
            result = await self._run_adb_command([
                "shell", "ping", "-c", "1", "8.8.8.8"
            ])
            
            return result['success'] and 'time=' in result.get('output', '')
            
        except Exception as e:
            logger.debug(f"Network connectivity check failed: {e}")
            return False

    async def _get_device_info(self) -> Dict[str, Any]:
        """Get device information for bug reports"""
        device_info = {}
        
        try:
            # Get device model
            model_result = await self._run_adb_command([
                "shell", "getprop", "ro.product.model"
            ])
            if model_result['success']:
                device_info['model'] = model_result['output'].strip()
            
            # Get Android version
            version_result = await self._run_adb_command([
                "shell", "getprop", "ro.build.version.release"
            ])
            if version_result['success']:
                device_info['android_version'] = version_result['output'].strip()
            
            # Get screen resolution
            size_result = await self._run_adb_command([
                "shell", "wm", "size"
            ])
            if size_result['success'] and "Physical size:" in size_result['output']:
                size_line = [l for l in size_result['output'].split('\n') if 'Physical size:' in l][0]
                device_info['screen_size'] = size_line.split(': ')[1].strip()
            
        except Exception as e:
            logger.debug(f"Device info collection failed: {e}")
        
        return device_info

    def _calculate_attention_cost(self, verification_report: VerificationReport) -> float:
        """Calculate attention cost for verification"""
        base_cost = 2.0
        
        # Add cost for number of bugs detected
        bug_cost = len(verification_report.bugs_detected) * 0.5
        
        # Add cost for verification time
        time_cost = min(verification_report.verification_time / 10.0, 2.0)
        
        # Add cost for strategies used
        strategy_cost = len(verification_report.learned_patterns.get('strategy_effectiveness', {})) * 0.3
        
        return base_cost + bug_cost + time_cost + strategy_cost

    async def _generate_verification_actions(self, verification_report: VerificationReport) -> List[str]:
        """Generate actions based on verification results"""
        actions = []
        
        # Add result summary
        actions.append(f"# Verification: {verification_report.verification_result.value}")
        actions.append(f"# Confidence: {verification_report.confidence_score:.2f}")
        actions.append(f"# Bugs detected: {len(verification_report.bugs_detected)}")
        
        # Add bug summaries
        for bug in verification_report.bugs_detected:
            actions.append(f"# Bug: {bug.title} ({bug.severity.value})")
        
        # Add recommendations
        if verification_report.verification_result == VerificationResult.FAIL:
            actions.append("# Recommendation: Investigate and fix detected issues")
        elif verification_report.verification_result == VerificationResult.PARTIAL:
            actions.append("# Recommendation: Review partial failures and improve reliability")
        else:
            actions.append("# Recommendation: Verification passed, continue with next steps")
        
        return actions

    def _update_verification_metrics(self, verification_report: VerificationReport) -> None:
        """Update verification performance metrics"""
        self.verification_metrics['total_verifications'] += 1
        
        if verification_report.verification_result == VerificationResult.PASS:
            self.verification_metrics['passed_verifications'] += 1
        elif verification_report.verification_result == VerificationResult.FAIL:
            self.verification_metrics['failed_verifications'] += 1
        
        self.verification_metrics['bugs_detected'] += len(verification_report.bugs_detected)
        
        # Update average verification time
        total = self.verification_metrics['total_verifications']
        current_avg = self.verification_metrics['average_verification_time']
        self.verification_metrics['average_verification_time'] = (
            (current_avg * (total - 1) + verification_report.verification_time) / total
        )

    # Analytics and reporting
    def get_verification_analytics(self) -> Dict[str, Any]:
        """Get comprehensive verification analytics"""
        return {
            'verification_metrics': self.verification_metrics.copy(),
            'strategy_effectiveness': {
                name: pattern.get('effectiveness_score', 0.0)
                for name, pattern in self.verification_patterns.items()
            },
            'bug_detection_summary': {
                'total_bugs_detected': len(self.bug_detection_history),
                'bug_types': self._analyze_bug_types(),
                'severity_distribution': self._analyze_bug_severity(),
                'false_positive_rate': len(self.false_positive_patterns) / max(1, len(self.bug_detection_history))
            },
            'learning_data': {
                'verification_patterns_learned': len(self.verification_patterns),
                'accuracy_tracking_points': len(self.verification_accuracy_tracking),
                'false_positive_patterns': len(self.false_positive_patterns)
            }
        }

    def _analyze_bug_types(self) -> Dict[str, int]:
        """Analyze distribution of bug types"""
        bug_types = {}
        for bug in self.bug_detection_history:
            bug_type = bug.bug_type.value
            bug_types[bug_type] = bug_types.get(bug_type, 0) + 1
        return bug_types

    def _analyze_bug_severity(self) -> Dict[str, int]:
        """Analyze distribution of bug severity"""
        severity_dist = {}
        for bug in self.bug_detection_history:
            severity = bug.severity.value
            severity_dist[severity] = severity_dist.get(severity, 0) + 1
        return severity_dist

    # Process task implementation
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process verification task"""
        instruction = task_data.get('instruction', 'Verify current state')
        observation = task_data.get('observation', {})
        context = task_data.get('context', {})
        
        reasoning_info, actions = await self.predict(instruction, observation, context)
        
        return {
            'task_id': task_data.get('task_id', 'unknown'),
            'reasoning_info': reasoning_info,
            'actions': actions,
            'verification_completed': True
        }


__all__ = [
    "VerifierAgent",
    "VerificationReport",
    "BugReport",
    "VerificationResult",
    "BugSeverity",
    "BugType"
]