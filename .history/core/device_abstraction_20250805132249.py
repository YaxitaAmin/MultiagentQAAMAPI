"""
Device Abstraction Layer - Universal Device Compatibility System
Provides cross-device compatibility and adaptation for Android automation
"""

import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from loguru import logger

from core.logger import AMAPILogger, LogCategory


class AndroidVersion(Enum):
    """Android version enumeration"""
    ANDROID_10 = "android_10"
    ANDROID_11 = "android_11"
    ANDROID_12 = "android_12"
    ANDROID_13 = "android_13"
    ANDROID_14 = "android_14"
    UNKNOWN = "unknown"


class DeviceClass(Enum):
    """Device performance class"""
    LOW_END = "low_end"
    MID_RANGE = "mid_range"
    HIGH_END = "high_end"
    FLAGSHIP = "flagship"


class AdaptationType(Enum):
    """Types of device adaptations"""
    COORDINATE_SCALING = "coordinate_scaling"
    UI_ELEMENT_MAPPING = "ui_element_mapping"
    TIMING_ADJUSTMENT = "timing_adjustment"
    GESTURE_MODIFICATION = "gesture_modification"
    ACCESSIBILITY_ENHANCEMENT = "accessibility_enhancement"


@dataclass
class DeviceFingerprint:
    """Unique device fingerprint"""
    device_id: str
    manufacturer: str
    model: str
    android_version: AndroidVersion
    api_level: int
    screen_width: int
    screen_height: int
    screen_density: int
    performance_class: str
    capabilities: List[str]
    ui_framework: str
    fingerprint_hash: str
    created_timestamp: float


@dataclass
class ActionTranslation:
    """Universal action translation result"""
    original_action: Dict[str, Any]
    translated_action: Dict[str, Any]
    adaptation_notes: List[str]
    compatibility_score: float
    success_probability: float
    alternative_actions: List[Dict[str, Any]]
    device_specific_params: Dict[str, Any]


@dataclass
class DeviceCapability:
    """Device capability descriptor"""
    capability_name: str
    supported: bool
    confidence: float
    version: Optional[str] = None
    limitations: List[str] = None
    alternatives: List[str] = None


class UniversalDeviceAbstraction:
    """
    Universal Device Abstraction Layer
    Provides cross-device compatibility for Android automation
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Device fingerprints and capabilities
        self.device_fingerprints: Dict[str, DeviceFingerprint] = {}
        self.device_capabilities: Dict[str, Dict[str, DeviceCapability]] = {}
        self.adaptation_rules: Dict[str, List[Dict[str, Any]]] = {}
        
        # Universal coordinate system
        self.reference_resolution = (1080, 1920)  # Standard reference
        self.reference_density = 420  # Standard DPI
        
        # Compatibility matrix
        self.compatibility_matrix: Dict[str, Dict[str, float]] = {}
        self.translation_cache: Dict[str, ActionTranslation] = {}
        
        # Performance tracking
        self.abstraction_metrics = {
            'devices_profiled': 0,
            'translations_performed': 0,
            'successful_adaptations': 0,
            'average_compatibility_score': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Logger
        self.logger = AMAPILogger("DeviceAbstraction")
        
        # Initialize standard device profiles
        self._initialize_standard_profiles()
        
        self.logger.info("Universal Device Abstraction initialized")

    def _initialize_standard_profiles(self):
        """Initialize standard device profiles and adaptation rules"""
        try:
            # Standard device profiles
            standard_devices = [
                {
                    'manufacturer': 'Google', 'model': 'Pixel_6',
                    'android_version': AndroidVersion.ANDROID_13,
                    'screen_width': 1080, 'screen_height': 2340,
                    'performance_class': 'high_end'
                },
                {
                    'manufacturer': 'Samsung', 'model': 'Galaxy_S22',
                    'android_version': AndroidVersion.ANDROID_12,
                    'screen_width': 1080, 'screen_height': 2340,
                    'performance_class': 'flagship'
                },
                {
                    'manufacturer': 'Generic', 'model': 'Emulator',
                    'android_version': AndroidVersion.ANDROID_11,
                    'screen_width': 1080, 'screen_height': 1920,
                    'performance_class': 'mid_range'
                }
            ]
            
            for device_info in standard_devices:
                fingerprint = self._create_device_fingerprint(device_info)
                self.device_fingerprints[fingerprint.device_id] = fingerprint
                self._initialize_device_capabilities(fingerprint)
            
            # Standard adaptation rules
            self._initialize_adaptation_rules()
            
            self.logger.debug(f"Initialized {len(standard_devices)} standard device profiles")
            
        except Exception as e:
            self.logger.error(f"Error initializing standard profiles: {e}")

    def _create_device_fingerprint(self, device_info: Dict[str, Any]) -> DeviceFingerprint:
        """Create device fingerprint from device information"""
        try:
            # Generate unique device ID
            device_id = f"{device_info['manufacturer']}_{device_info['model']}_{int(time.time())}"
            
            # Calculate fingerprint hash
            fingerprint_data = (
                f"{device_info['manufacturer']}{device_info['model']}"
                f"{device_info['android_version'].value}"
                f"{device_info['screen_width']}x{device_info['screen_height']}"
            )
            fingerprint_hash = str(hash(fingerprint_data))
            
            return DeviceFingerprint(
                device_id=device_id,
                manufacturer=device_info['manufacturer'],
                model=device_info['model'],
                android_version=device_info['android_version'],
                api_level=device_info.get('api_level', 30),
                screen_width=device_info['screen_width'],
                screen_height=device_info['screen_height'],
                screen_density=device_info.get('screen_density', 420),
                performance_class=device_info['performance_class'],
                capabilities=device_info.get('capabilities', []),
                ui_framework=device_info.get('ui_framework', 'android_ui'),
                fingerprint_hash=fingerprint_hash,
                created_timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error creating device fingerprint: {e}")
            raise

    def _initialize_device_capabilities(self, fingerprint: DeviceFingerprint):
        """Initialize capabilities for a device"""
        try:
            capabilities = {}
            
            # Standard Android capabilities
            standard_capabilities = [
                'touch_input', 'swipe_gestures', 'text_input',
                'screenshot', 'ui_automation', 'accessibility_service'
            ]
            
            for capability in standard_capabilities:
                capabilities[capability] = DeviceCapability(
                    capability_name=capability,
                    supported=True,
                    confidence=0.95,
                    limitations=[],
                    alternatives=[]
                )
            
            # Version-specific capabilities
            if fingerprint.android_version in [AndroidVersion.ANDROID_12, AndroidVersion.ANDROID_13, AndroidVersion.ANDROID_14]:
                capabilities['material_you'] = DeviceCapability(
                    capability_name='material_you',
                    supported=True,
                    confidence=0.9
                )
            
            # Performance-specific capabilities
            if fingerprint.performance_class in ['high_end', 'flagship']:
                capabilities['high_refresh_rate'] = DeviceCapability(
                    capability_name='high_refresh_rate',
                    supported=True,
                    confidence=0.8
                )
            
            self.device_capabilities[fingerprint.device_id] = capabilities
            
        except Exception as e:
            self.logger.error(f"Error initializing device capabilities: {e}")

    def _initialize_adaptation_rules(self):
        """Initialize standard adaptation rules"""
        try:
            # Coordinate scaling rules
            self.adaptation_rules['coordinate_scaling'] = [
                {
                    'condition': 'resolution_mismatch',
                    'action': 'scale_coordinates',
                    'parameters': {'method': 'proportional'}
                }
            ]
            
            # UI element mapping rules
            self.adaptation_rules['ui_element_mapping'] = [
                {
                    'condition': 'element_not_found',
                    'action': 'search_alternatives',
                    'parameters': {'search_radius': 50, 'similarity_threshold': 0.8}
                }
            ]
            
            # Timing adjustment rules
            self.adaptation_rules['timing_adjustment'] = [
                {
                    'condition': 'low_performance_device',
                    'action': 'increase_timeouts',
                    'parameters': {'multiplier': 1.5}
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Error initializing adaptation rules: {e}")

    async def detect_device_fingerprint(self, device_info: Dict[str, Any]) -> DeviceFingerprint:
        """Detect and create device fingerprint from device information"""
        try:
            # Check if we already have this device
            fingerprint_hash = self._calculate_fingerprint_hash(device_info)
            
            for fingerprint in self.device_fingerprints.values():
                if fingerprint.fingerprint_hash == fingerprint_hash:
                    self.logger.debug(f"Device fingerprint found in cache: {fingerprint.device_id}")
                    return fingerprint
            
            # Create new fingerprint
            android_version = self._detect_android_version(device_info)
            performance_class = self._detect_performance_class(device_info)
            
            enhanced_device_info = {
                **device_info,
                'android_version': android_version,
                'performance_class': performance_class
            }
            
            fingerprint = self._create_device_fingerprint(enhanced_device_info)
            
            # Store fingerprint and initialize capabilities
            self.device_fingerprints[fingerprint.device_id] = fingerprint
            self._initialize_device_capabilities(fingerprint)
            
            self.abstraction_metrics['devices_profiled'] += 1
            
            self.logger.info(f"New device fingerprint created: {fingerprint.manufacturer} {fingerprint.model}")
            
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Error detecting device fingerprint: {e}")
            raise

    def _calculate_fingerprint_hash(self, device_info: Dict[str, Any]) -> str:
        """Calculate fingerprint hash from device info"""
        try:
            fingerprint_data = (
                f"{device_info.get('manufacturer', 'Unknown')}"
                f"{device_info.get('model', 'Unknown')}"
                f"{device_info.get('api_level', 30)}"
                f"{device_info.get('screen_width', 1080)}x{device_info.get('screen_height', 1920)}"
            )
            return str(hash(fingerprint_data))
            
        except Exception as e:
            self.logger.error(f"Error calculating fingerprint hash: {e}")
            return str(hash("unknown"))

    def _detect_android_version(self, device_info: Dict[str, Any]) -> AndroidVersion:
        """Detect Android version from device info"""
        try:
            api_level = device_info.get('api_level', 30)
            
            version_map = {
                29: AndroidVersion.ANDROID_10,
                30: AndroidVersion.ANDROID_11,
                31: AndroidVersion.ANDROID_12,
                32: AndroidVersion.ANDROID_12,
                33: AndroidVersion.ANDROID_13,
                34: AndroidVersion.ANDROID_14
            }
            
            return version_map.get(api_level, AndroidVersion.UNKNOWN)
            
        except Exception as e:
            self.logger.error(f"Error detecting Android version: {e}")
            return AndroidVersion.UNKNOWN

    def _detect_performance_class(self, device_info: Dict[str, Any]) -> str:
        """Detect device performance class"""
        try:
            # Simple heuristic based on screen resolution and manufacturer
            width = device_info.get('screen_width', 1080)
            height = device_info.get('screen_height', 1920)
            manufacturer = device_info.get('manufacturer', '').lower()
            
            total_pixels = width * height
            
            if total_pixels > 2400000:  # > 1440p
                return 'flagship'
            elif total_pixels > 2000000:  # > 1080p
                if manufacturer in ['google', 'samsung', 'oneplus']:
                    return 'high_end'
                else:
                    return 'mid_range'
            else:
                return 'low_end'
                
        except Exception as e:
            self.logger.error(f"Error detecting performance class: {e}")
            return 'mid_range'

    async def translate_action_universally(self, action: Dict[str, Any], 
                                         target_device: DeviceFingerprint) -> ActionTranslation:
        """Translate action for universal device compatibility"""
        try:
            # Check cache first
            cache_key = f"{hash(str(action))}_{target_device.device_id}"
            if cache_key in self.translation_cache:
                self.abstraction_metrics['cache_hit_rate'] = (
                    (self.abstraction_metrics['cache_hit_rate'] * self.abstraction_metrics['translations_performed'] + 1) /
                    (self.abstraction_metrics['translations_performed'] + 1)
                )
                return self.translation_cache[cache_key]
            
            # Perform translation
            translation = await self._perform_action_translation(action, target_device)
            
            # Cache result
            self.translation_cache[cache_key] = translation
            
            # Update metrics
            self.abstraction_metrics['translations_performed'] += 1
            if translation.compatibility_score > 0.8:
                self.abstraction_metrics['successful_adaptations'] += 1
            
            # Update average compatibility score
            total_translations = self.abstraction_metrics['translations_performed']
            current_avg = self.abstraction_metrics['average_compatibility_score']
            self.abstraction_metrics['average_compatibility_score'] = (
                (current_avg * (total_translations - 1) + translation.compatibility_score) / total_translations
            )
            
            return translation
            
        except Exception as e:
            self.logger.error(f"Error translating action universally: {e}")
            # Return fallback translation
            return ActionTranslation(
                original_action=action,
                translated_action=action,
                adaptation_notes=[f"Translation failed: {str(e)}"],
                compatibility_score=0.5,
                success_probability=0.5,
                alternative_actions=[],
                device_specific_params={}
            )

    async def _perform_action_translation(self, action: Dict[str, Any], 
                                        target_device: DeviceFingerprint) -> ActionTranslation:
        """Perform the actual action translation"""
        try:
            translated_action = action.copy()
            adaptation_notes = []
            compatibility_score = 1.0
            success_probability = 0.9
            alternative_actions = []
            device_specific_params = {}
            
            action_type = action.get('type', action.get('action_type', 'unknown'))
            
            # Apply coordinate scaling if needed
            if action_type in ['tap', 'click', 'swipe'] and 'coordinates' in action:
                scaled_coords, scaling_notes = self._scale_coordinates(
                    action['coordinates'], target_device
                )
                translated_action['coordinates'] = scaled_coords
                adaptation_notes.extend(scaling_notes)
                
                if scaling_notes:
                    compatibility_score *= 0.95  # Small penalty for scaling
            
            # Apply timing adjustments
            if target_device.performance_class == 'low_end':
                timing_adjustments = self._apply_timing_adjustments(translated_action, target_device)
                translated_action.update(timing_adjustments)
                adaptation_notes.append("Applied timing adjustments for low-end device")
                compatibility_score *= 0.98
            
            # Apply UI framework adaptations
            ui_adaptations = self._apply_ui_framework_adaptations(translated_action, target_device)
            translated_action.update(ui_adaptations)
            
            # Generate alternative actions if compatibility is low
            if compatibility_score < 0.8:
                alternative_actions = self._generate_alternative_actions(action, target_device)
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(
                action, translated_action, target_device, compatibility_score
            )
            
            return ActionTranslation(
                original_action=action,
                translated_action=translated_action,
                adaptation_notes=adaptation_notes,
                compatibility_score=compatibility_score,
                success_probability=success_probability,
                alternative_actions=alternative_actions,
                device_specific_params=device_specific_params
            )
            
        except Exception as e:
            self.logger.error(f"Error performing action translation: {e}")
            raise

    def _scale_coordinates(self, coordinates: Union[Tuple[int, int], List[int]], 
                          target_device: DeviceFingerprint) -> Tuple[Union[Tuple[int, int], List[int]], List[str]]:
        """Scale coordinates for target device"""
        try:
            notes = []
            
            # Get scaling factors
            width_scale = target_device.screen_width / self.reference_resolution[0]
            height_scale = target_device.screen_height / self.reference_resolution[1]
            
            if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
                # Single coordinate pair
                scaled_x = int(coordinates[0] * width_scale)
                scaled_y = int(coordinates[1] * height_scale)
                
                if width_scale != 1.0 or height_scale != 1.0:
                    notes.append(f"Scaled coordinates from {coordinates} to ({scaled_x}, {scaled_y})")
                
                return (scaled_x, scaled_y), notes
            
            elif isinstance(coordinates, (list, tuple)) and len(coordinates) == 4:
                # Rectangle or swipe coordinates
                scaled_coords = [
                    int(coordinates[0] * width_scale),  # x1
                    int(coordinates[1] * height_scale), # y1
                    int(coordinates[2] * width_scale),  # x2
                    int(coordinates[3] * height_scale)  # y2
                ]
                
                if width_scale != 1.0 or height_scale != 1.0:
                    notes.append(f"Scaled rectangle from {coordinates} to {scaled_coords}")
                
                return scaled_coords, notes
            
            else:
                return coordinates, notes
            
        except Exception as e:
            self.logger.error(f"Error scaling coordinates: {e}")
            return coordinates, [f"Coordinate scaling failed: {str(e)}"]

    def _apply_timing_adjustments(self, action: Dict[str, Any], 
                                target_device: DeviceFingerprint) -> Dict[str, Any]:
        """Apply timing adjustments based on device performance"""
        try:
            adjustments = {}
            
            # Performance-based multipliers
            multipliers = {
                'low_end': 2.0,
                'mid_range': 1.3,
                'high_end': 1.0,
                'flagship': 0.8
            }
            
            multiplier = multipliers.get(target_device.performance_class, 1.0)
            
            # Adjust various timing parameters
            if 'wait_time' in action:
                adjustments['wait_time'] = action['wait_time'] * multiplier
            
            if 'timeout' in action:
                adjustments['timeout'] = action['timeout'] * multiplier
            
            if 'delay' in action:
                adjustments['delay'] = action['delay'] * multiplier
            
            # Add default wait for low-end devices
            if target_device.performance_class == 'low_end' and 'wait_time' not in action:
                adjustments['wait_time'] = 1.0
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error applying timing adjustments: {e}")
            return {}

    def _apply_ui_framework_adaptations(self, action: Dict[str, Any], 
                                      target_device: DeviceFingerprint) -> Dict[str, Any]:
        """Apply UI framework specific adaptations"""
        try:
            adaptations = {}
            
            # Android version specific adaptations
            if target_device.android_version == AndroidVersion.ANDROID_14:
                # Material You adaptations
                if 'ui_element' in action:
                    adaptations['material_you_compatible'] = True
            
            # Manufacturer specific adaptations
            if target_device.manufacturer.lower() == 'samsung':
                # Samsung One UI adaptations
                if 'gesture' in action:
                    adaptations['one_ui_gesture'] = True
            
            return adaptations
            
        except Exception as e:
            self.logger.error(f"Error applying UI framework adaptations: {e}")
            return {}

    def _generate_alternative_actions(self, original_action: Dict[str, Any], 
                                    target_device: DeviceFingerprint) -> List[Dict[str, Any]]:
        """Generate alternative actions for better compatibility"""
        try:
            alternatives = []
            action_type = original_action.get('type', original_action.get('action_type', 'unknown'))
            
            if action_type == 'tap':
                # Alternative: Long press
                alternatives.append({
                    **original_action,
                    'type': 'long_press',
                    'duration': 1000
                })
                
                # Alternative: Double tap
                alternatives.append({
                    **original_action,
                    'type': 'double_tap',
                    'interval': 100
                })
            
            elif action_type == 'swipe':
                # Alternative: Multi-step swipe
                if 'coordinates' in original_action and len(original_action['coordinates']) == 4:
                    coords = original_action['coordinates']
                    mid_x = (coords[0] + coords[2]) // 2
                    mid_y = (coords[1] + coords[3]) // 2
                    
                    alternatives.append({
                        'type': 'multi_swipe',
                        'steps': [
                            {'coordinates': [coords[0], coords[1], mid_x, mid_y]},
                            {'coordinates': [mid_x, mid_y, coords[2], coords[3]]}
                        ]
                    })
            
            return alternatives[:3]  # Limit to 3 alternatives
            
        except Exception as e:
            self.logger.error(f"Error generating alternative actions: {e}")
            return []

    def _calculate_success_probability(self, original_action: Dict[str, Any],
                                     translated_action: Dict[str, Any],
                                     target_device: DeviceFingerprint,
                                     compatibility_score: float) -> float:
        """Calculate success probability for translated action"""
        try:
            base_probability = 0.8
            
            # Adjust based on compatibility score
            compatibility_factor = compatibility_score
            
            # Adjust based on device capabilities
            device_capabilities = self.device_capabilities.get(target_device.device_id, {})
            action_type = original_action.get('type', original_action.get('action_type', 'unknown'))
            
            if action_type in device_capabilities:
                capability = device_capabilities[action_type]
                if capability.supported:
                    capability_factor = capability.confidence
                else:
                    capability_factor = 0.3  # Low probability if not supported
            else:
                capability_factor = 0.7  # Default for unknown capabilities
            
            # Adjust based on device performance
            performance_factors = {
                'low_end': 0.85,
                'mid_range': 0.9,
                'high_end': 0.95,
                'flagship': 0.98
            }
            performance_factor = performance_factors.get(target_device.performance_class, 0.9)
            
            # Calculate final probability
            success_probability = (
                base_probability * 0.3 +
                compatibility_factor * 0.4 +
                capability_factor * 0.2 +
                performance_factor * 0.1
            )
            
            return min(1.0, max(0.1, success_probability))
            
        except Exception as e:
            self.logger.error(f"Error calculating success probability: {e}")
            return 0.5

    def get_device_compatibility_matrix(self) -> Dict[str, Any]:
        """Get compatibility matrix between devices"""
        try:
            matrix = {}
            
            device_ids = list(self.device_fingerprints.keys())
            
            for device1_id in device_ids:
                matrix[device1_id] = {}
                device1 = self.device_fingerprints[device1_id]
                
                for device2_id in device_ids:
                    device2 = self.device_fingerprints[device2_id]
                    
                    if device1_id == device2_id:
                        compatibility = 1.0
                    else:
                        compatibility = self._calculate_device_compatibility(device1, device2)
                    
                    matrix[device1_id][device2_id] = compatibility
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error generating compatibility matrix: {e}")
            return {}

    def _calculate_device_compatibility(self, device1: DeviceFingerprint, 
                                      device2: DeviceFingerprint) -> float:
        """Calculate compatibility between two devices"""
        try:
            compatibility_factors = []
            
            # Screen resolution similarity
            res1 = (device1.screen_width, device1.screen_height)
            res2 = (device2.screen_width, device2.screen_height)
            
            res_similarity = 1.0 - abs(
                (res1[0] * res1[1] - res2[0] * res2[1]) / 
                max(res1[0] * res1[1], res2[0] * res2[1])
            )
            compatibility_factors.append(res_similarity * 0.3)
            
            # Android version similarity
            version_similarity = 1.0 if device1.android_version == device2.android_version else 0.8
            compatibility_factors.append(version_similarity * 0.2)
            
            # Performance class similarity
            perf_classes = ['low_end', 'mid_range', 'high_end', 'flagship']
            perf1_idx = perf_classes.index(device1.performance_class) if device1.performance_class in perf_classes else 1
            perf2_idx = perf_classes.index(device2.performance_class) if device2.performance_class in perf_classes else 1
            
            perf_similarity = 1.0 - abs(perf1_idx - perf2_idx) / len(perf_classes)
            compatibility_factors.append(perf_similarity * 0.2)
            
            # Manufacturer similarity
            manufacturer_similarity = 1.0 if device1.manufacturer == device2.manufacturer else 0.7
            compatibility_factors.append(manufacturer_similarity * 0.1)
            
            # UI framework similarity
            ui_similarity = 1.0 if device1.ui_framework == device2.ui_framework else 0.8
            compatibility_factors.append(ui_similarity * 0.1)
            
            # Capabilities similarity
            caps1 = set(device1.capabilities)
            caps2 = set(device2.capabilities)
            
            if caps1 or caps2:
                caps_similarity = len(caps1 & caps2) / len(caps1 | caps2) if caps1 | caps2 else 1.0
            else:
                caps_similarity = 1.0
            
            compatibility_factors.append(caps_similarity * 0.1)
            
            return sum(compatibility_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating device compatibility: {e}")
            return 0.5

    def get_abstraction_analytics(self) -> Dict[str, Any]:
        """Get comprehensive abstraction analytics"""
        try:
            # Device statistics
            device_stats = {
                'total_devices': len(self.device_fingerprints),
                'manufacturers': len(set(d.manufacturer for d in self.device_fingerprints.values())),
                'android_versions': len(set(d.android_version for d in self.device_fingerprints.values())),
                'performance_classes': len(set(d.performance_class for d in self.device_fingerprints.values()))
            }
            
            # Translation statistics
            cache_stats = {
                'cache_size': len(self.translation_cache),
                'cache_hit_rate': self.abstraction_metrics['cache_hit_rate']
            }
            
            # Compatibility analysis
            compatibility_matrix = self.get_device_compatibility_matrix()
            if compatibility_matrix:
                all_scores = []
                for device1_scores in compatibility_matrix.values():
                    all_scores.extend([score for score in device1_scores.values() if score < 1.0])
                
                avg_compatibility = np.mean(all_scores) if all_scores else 1.0
            else:
                avg_compatibility = 1.0
            
            return {
                'abstraction_metrics': self.abstraction_metrics.copy(),
                'device_statistics': device_stats,
                'cache_statistics': cache_stats,
                'average_device_compatibility': avg_compatibility,
                'supported_adaptations': list(self.adaptation_rules.keys()),
                'total_adaptation_rules': sum(len(rules) for rules in self.adaptation_rules.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error generating abstraction analytics: {e}")
            return {'error': str(e)}

    def get_device_profile(self, device_id: str) -> Dict[str, Any]:
        """Get detailed profile for specific device"""
        try:
            if device_id not in self.device_fingerprints:
                return {'error': 'Device not found'}
            
            fingerprint = self.device_fingerprints[device_id]
            capabilities = self.device_capabilities.get(device_id, {})
            
            return {
                'fingerprint': asdict(fingerprint),
                'capabilities': {
                    name: asdict(cap) for name, cap in capabilities.items()
                },
                'supported_adaptations': self._get_supported_adaptations(fingerprint),
                'compatibility_scores': self.compatibility_matrix.get(device_id, {}),
                'translation_cache_entries': sum(
                    1 for key in self.translation_cache.keys() 
                    if device_id in key
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting device profile: {e}")
            return {'error': str(e)}

    def _get_supported_adaptations(self, fingerprint: DeviceFingerprint) -> List[str]:
        """Get list of supported adaptations for device"""
        try:
            supported = []
            
            # All devices support coordinate scaling
            supported.append('coordinate_scaling')
            
            # Performance-based adaptations
            if fingerprint.performance_class in ['low_end', 'mid_range']:
                supported.append('timing_adjustment')
            
            # Version-based adaptations
            if fingerprint.android_version in [AndroidVersion.ANDROID_12, AndroidVersion.ANDROID_13, AndroidVersion.ANDROID_14]:
                supported.append('material_design_adaptation')
            
            # UI framework adaptations
            if fingerprint.ui_framework == 'android_ui':
                supported.append('ui_element_mapping')
            
            return supported
            
        except Exception as e:
            self.logger.error(f"Error getting supported adaptations: {e}")
            return []

    async def optimize_for_device(self, device_id: str) -> Dict[str, Any]:
        """Optimize abstraction layer for specific device"""
        try:
            if device_id not in self.device_fingerprints:
                return {'success': False, 'error': 'Device not found'}
            
            fingerprint = self.device_fingerprints[device_id]
            
            # Create device-specific optimization rules
            optimizations = []
            
            # Screen-specific optimizations
            if fingerprint.screen_width > 1440:
                optimizations.append({
                    'type': 'high_resolution_optimization',
                    'parameters': {'coordinate_precision': 'high'}
                })
            
            # Performance-specific optimizations
            if fingerprint.performance_class == 'low_end':
                optimizations.append({
                    'type': 'performance_optimization',
                    'parameters': {'reduce_animations': True, 'increase_timeouts': True}
                })
            
            return {
                'success': True,
                'device_id': device_id,
                'optimizations_applied': len(optimizations),
                'optimizations': optimizations
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing for device: {e}")
            return {'success': False, 'error': str(e)}


__all__ = [
    "UniversalDeviceAbstraction",
    "DeviceFingerprint",
    "ActionTranslation",
    "DeviceCapability",
    "AndroidVersion",
    "DeviceClass",
    "AdaptationType"
]