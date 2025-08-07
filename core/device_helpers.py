"""
Device Utilities - Helper functions for device operations and compatibility
Cross-platform device interaction utilities for AMAPI system
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

"""
Quick fix for import errors in the AMAPI system
This file provides the missing imports and fixes
"""

# Fix 1: Add DeviceCompatibilityChecker to utils/device_helpers.py
from device_compatibility_checker import DeviceCompatibilityChecker, DeviceProfile, CompatibilityLevel

# Fix 2: Update AttentionPoolType with missing values
from .attention_economics import AttentionPoolType, AttentionEconomicsEngine

@dataclass
class ScreenCoordinate:
    """Screen coordinate with device context"""
    x: float
    y: float
    reference_width: int = 1080
    reference_height: int = 1920
    confidence: float = 1.0


@dataclass
class GestureParameters:
    """Parameters for gesture operations"""
    start_point: ScreenCoordinate
    end_point: Optional[ScreenCoordinate] = None
    duration: float = 0.3
    pressure: float = 1.0
    gesture_type: str = "tap"


@dataclass
class DeviceCapabilities:
    """Device capabilities and constraints"""
    supports_multitouch: bool = True
    supports_pressure: bool = False
    supports_gestures: bool = True
    max_simultaneous_touches: int = 10
    screen_refresh_rate: int = 60
    input_latency_ms: float = 16.0


def normalize_coordinates(x: float, y: float, source_resolution: Tuple[int, int],
                         target_resolution: Tuple[int, int]) -> Tuple[float, float]:
    """
    Normalize coordinates between different screen resolutions
    
    Args:
        x, y: Source coordinates
        source_resolution: (width, height) of source device
        target_resolution: (width, height) of target device
        
    Returns:
        Normalized coordinates for target device
    """
    try:
        source_width, source_height = source_resolution
        target_width, target_height = target_resolution
        
        # Calculate scaling factors
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        
        # Apply scaling
        normalized_x = x * scale_x
        normalized_y = y * scale_y
        
        # Ensure coordinates are within bounds
        normalized_x = max(0, min(target_width - 1, normalized_x))
        normalized_y = max(0, min(target_height - 1, normalized_y))
        
        return (normalized_x, normalized_y)
        
    except Exception:
        return (x, y)


def calculate_density_scaling(source_density: float, target_density: float) -> float:
    """
    Calculate scaling factor based on screen density
    
    Args:
        source_density: Source device DPI
        target_density: Target device DPI
        
    Returns:
        Density scaling factor
    """
    try:
        if source_density <= 0 or target_density <= 0:
            return 1.0
        
        scaling_factor = target_density / source_density
        
        # Clamp to reasonable range
        return max(0.5, min(3.0, scaling_factor))
        
    except Exception:
        return 1.0


def adapt_gesture_for_device(gesture: GestureParameters, device_capabilities: DeviceCapabilities,
                           device_resolution: Tuple[int, int]) -> GestureParameters:
    """
    Adapt gesture parameters for specific device capabilities
    
    Args:
        gesture: Original gesture parameters
        device_capabilities: Target device capabilities
        device_resolution: Target device resolution
        
    Returns:
        Adapted gesture parameters
    """
    try:
        adapted_gesture = GestureParameters(
            start_point=gesture.start_point,
            end_point=gesture.end_point,
            duration=gesture.duration,
            pressure=gesture.pressure,
            gesture_type=gesture.gesture_type
        )
        
        # Adapt coordinates to device resolution
        if gesture.start_point:
            start_x, start_y = normalize_coordinates(
                gesture.start_point.x, gesture.start_point.y,
                (gesture.start_point.reference_width, gesture.start_point.reference_height),
                device_resolution
            )
            adapted_gesture.start_point = ScreenCoordinate(
                x=start_x, y=start_y,
                reference_width=device_resolution[0],
                reference_height=device_resolution[1],
                confidence=gesture.start_point.confidence
            )
        
        if gesture.end_point:
            end_x, end_y = normalize_coordinates(
                gesture.end_point.x, gesture.end_point.y,
                (gesture.end_point.reference_width, gesture.end_point.reference_height),
                device_resolution
            )
            adapted_gesture.end_point = ScreenCoordinate(
                x=end_x, y=end_y,
                reference_width=device_resolution[0],
                reference_height=device_resolution[1],
                confidence=gesture.end_point.confidence
            )
        
        # Adapt duration based on device performance
        latency_compensation = device_capabilities.input_latency_ms / 16.0  # Normalize to 16ms baseline
        adapted_gesture.duration = max(0.1, gesture.duration * latency_compensation)
        
        # Adapt pressure if not supported
        if not device_capabilities.supports_pressure:
            adapted_gesture.pressure = 1.0
        
        return adapted_gesture
        
    except Exception:
        return gesture


def calculate_touch_target_size(base_size: int, device_density: float, 
                              accessibility_scaling: float = 1.0) -> int:
    """
    Calculate appropriate touch target size for device
    
    Args:
        base_size: Base touch target size in pixels
        device_density: Device screen density (DPI)
        accessibility_scaling: User accessibility scaling factor
        
    Returns:
        Adapted touch target size
    """
    try:
        # Standard density baseline (160 DPI for Android)
        standard_density = 160.0
        
        # Calculate density scaling
        density_scale = device_density / standard_density
        
        # Apply scaling factors
        adapted_size = base_size * density_scale * accessibility_scaling
        
        # Ensure minimum touch target size (48dp minimum for Android, ~9mm)
        min_size = max(32, int(48 * density_scale))
        
        return max(min_size, int(adapted_size))
        
    except Exception:
        return base_size


def generate_swipe_path(start: ScreenCoordinate, end: ScreenCoordinate,
                       steps: int = 10) -> List[ScreenCoordinate]:
    """
    Generate intermediate points for smooth swipe gesture
    
    Args:
        start: Starting coordinate
        end: Ending coordinate
        steps: Number of intermediate steps
        
    Returns:
        List of coordinates for smooth swipe path
    """
    try:
        if steps <= 1:
            return [start, end]
        
        path = []
        
        for i in range(steps + 1):
            progress = i / steps
            
            # Linear interpolation with slight curve for natural movement
            curve_factor = math.sin(progress * math.pi) * 0.1
            
            x = start.x + (end.x - start.x) * progress
            y = start.y + (end.y - start.y) * progress + curve_factor * abs(end.x - start.x) * 0.1
            
            coordinate = ScreenCoordinate(
                x=x, y=y,
                reference_width=start.reference_width,
                reference_height=start.reference_height,
                confidence=min(start.confidence, end.confidence)
            )
            
            path.append(coordinate)
        
        return path
        
    except Exception:
        return [start, end]


def calculate_gesture_velocity(distance: float, duration: float) -> float:
    """
    Calculate gesture velocity for natural movement
    
    Args:
        distance: Distance of gesture in pixels
        duration: Duration of gesture in seconds
        
    Returns:
        Velocity in pixels per second
    """
    try:
        if duration <= 0:
            return 0.0
        
        velocity = distance / duration
        
        # Clamp to realistic human gesture speeds
        min_velocity = 50.0   # pixels/second
        max_velocity = 2000.0  # pixels/second
        
        return max(min_velocity, min(max_velocity, velocity))
        
    except Exception:
        return 500.0


def detect_gesture_type(start: ScreenCoordinate, end: Optional[ScreenCoordinate] = None,
                       duration: float = 0.3) -> str:
    """
    Detect appropriate gesture type based on parameters
    
    Args:
        start: Starting coordinate
        end: Optional ending coordinate
        duration: Gesture duration
        
    Returns:
        Detected gesture type
    """
    try:
        if not end:
            # Single point gesture
            if duration < 0.2:
                return "tap"
            elif duration < 1.0:
                return "long_press"
            else:
                return "extended_press"
        
        # Calculate distance
        distance = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
        
        if distance < 20:
            # Very small movement - treat as tap
            return "tap"
        elif distance < 100:
            # Small movement
            if duration < 0.5:
                return "short_swipe"
            else:
                return "drag"
        else:
            # Large movement
            if duration < 0.8:
                return "swipe"
            else:
                return "long_drag"
                
    except Exception:
        return "tap"


def optimize_gesture_timing(gesture_type: str, distance: float = 0.0,
                          device_capabilities: Optional[DeviceCapabilities] = None) -> float:
    """
    Optimize gesture timing based on type and device capabilities
    
    Args:
        gesture_type: Type of gesture
        distance: Distance of gesture movement
        device_capabilities: Device capabilities
        
    Returns:
        Optimized duration in seconds
    """
    try:
        # Base durations for different gesture types
        base_durations = {
            "tap": 0.1,
            "long_press": 1.0,
            "extended_press": 2.0,
            "short_swipe": 0.3,
            "swipe": 0.5,
            "drag": 0.8,
            "long_drag": 1.2
        }
        
        base_duration = base_durations.get(gesture_type, 0.5)
        
        # Adjust for distance
        if distance > 0 and gesture_type in ["swipe", "drag", "long_drag", "short_swipe"]:
            # Calculate duration based on natural velocity
            natural_velocity = 800.0  # pixels per second
            distance_duration = distance / natural_velocity
            base_duration = max(base_duration, distance_duration)
        
        # Adjust for device capabilities
        if device_capabilities:
            # Slower devices need more time
            latency_factor = device_capabilities.input_latency_ms / 16.0
            refresh_factor = 60.0 / device_capabilities.screen_refresh_rate
            
            device_factor = max(1.0, latency_factor * refresh_factor)
            base_duration *= device_factor
        
        # Ensure reasonable bounds
        return max(0.05, min(5.0, base_duration))
        
    except Exception:
        return 0.5


def validate_coordinates(coordinates: ScreenCoordinate, device_resolution: Tuple[int, int]) -> bool:
    """
    Validate that coordinates are within device screen bounds
    
    Args:
        coordinates: Coordinates to validate
        device_resolution: Device screen resolution (width, height)
        
    Returns:
        True if coordinates are valid
    """
    try:
        width, height = device_resolution
        
        return (0 <= coordinates.x < width and 
                0 <= coordinates.y < height and
                coordinates.confidence > 0.0)
        
    except Exception:
        return False


def calculate_safe_gesture_area(device_resolution: Tuple[int, int], 
                              safe_margin: int = 50) -> Dict[str, int]:
    """
    Calculate safe area for gestures avoiding screen edges
    
    Args:
        device_resolution: Device screen resolution
        safe_margin: Margin from screen edges in pixels
        
    Returns:
        Dictionary with safe area bounds
    """
    try:
        width, height = device_resolution
        
        return {
            'left': safe_margin,
            'top': safe_margin,
            'right': width - safe_margin,
            'bottom': height - safe_margin,
            'center_x': width // 2,
            'center_y': height // 2
        }
        
    except Exception:
        return {
            'left': 50, 'top': 50, 'right': 1030, 'bottom': 1870,
            'center_x': 540, 'center_y': 960
        }


def estimate_gesture_success_probability(gesture: GestureParameters,
                                       device_capabilities: DeviceCapabilities,
                                       ui_element_size: Optional[int] = None) -> float:
    """
    Estimate probability of gesture success
    
    Args:
        gesture: Gesture parameters
        device_capabilities: Device capabilities
        ui_element_size: Size of target UI element
        
    Returns:
        Success probability (0.0 to 1.0)
    """
    try:
        base_probability = 0.8
        
        # Coordinate confidence factor
        coord_confidence = gesture.start_point.confidence
        if gesture.end_point:
            coord_confidence = min(coord_confidence, gesture.end_point.confidence)
        
        # Target size factor
        if ui_element_size:
            min_target_size = calculate_touch_target_size(48, 420)  # Standard minimum
            if ui_element_size < min_target_size:
                size_penalty = (min_target_size - ui_element_size) / min_target_size * 0.3
                base_probability -= size_penalty
        
        # Device capability factor
        capability_bonus = 0.0
        if device_capabilities.supports_multitouch and gesture.gesture_type in ["swipe", "drag"]:
            capability_bonus += 0.05
        if device_capabilities.input_latency_ms <= 20:
            capability_bonus += 0.1
        
        # Gesture complexity factor
        complexity_penalty = 0.0
        if gesture.gesture_type in ["long_drag", "extended_press"]:
            complexity_penalty = 0.1
        
        final_probability = base_probability * coord_confidence + capability_bonus - complexity_penalty
        
        return max(0.1, min(1.0, final_probability))
        
    except Exception:
        return 0.7


def create_fallback_gesture(original_gesture: GestureParameters,
                          device_resolution: Tuple[int, int]) -> GestureParameters:
    """
    Create fallback gesture for failed attempts
    
    Args:
        original_gesture: Original gesture that failed
        device_resolution: Device screen resolution
        
    Returns:
        Fallback gesture parameters
    """
    try:
        safe_area = calculate_safe_gesture_area(device_resolution)
        
        # For complex gestures, fallback to simple tap
        if original_gesture.gesture_type in ["long_drag", "extended_press"]:
            fallback_type = "tap"
            fallback_duration = 0.2
        else:
            fallback_type = original_gesture.gesture_type
            fallback_duration = min(1.0, original_gesture.duration * 1.2)
        
        # Use center of safe area if original coordinates seem problematic
        fallback_x = safe_area['center_x']
        fallback_y = safe_area['center_y']
        
        # If original coordinates were valid, use them with small offset
        if validate_coordinates(original_gesture.start_point, device_resolution):
            fallback_x = max(safe_area['left'], 
                           min(safe_area['right'], original_gesture.start_point.x + 10))
            fallback_y = max(safe_area['top'], 
                           min(safe_area['bottom'], original_gesture.start_point.y + 10))
        
        fallback_start = ScreenCoordinate(
            x=fallback_x, y=fallback_y,
            reference_width=device_resolution[0],
            reference_height=device_resolution[1],
            confidence=0.9
        )
        
        return GestureParameters(
            start_point=fallback_start,
            end_point=None,
            duration=fallback_duration,
            pressure=1.0,
            gesture_type=fallback_type
        )
        
    except Exception:
        # Ultimate fallback - center screen tap
        return GestureParameters(
            start_point=ScreenCoordinate(x=540, y=960, confidence=0.8),
            duration=0.3,
            gesture_type="tap"
        )


__all__ = [
    "ScreenCoordinate",
    "GestureParameters", 
    "DeviceCapabilities",
    "normalize_coordinates",
    "calculate_density_scaling",
    "adapt_gesture_for_device",
    "calculate_touch_target_size",
    "generate_swipe_path",
    "calculate_gesture_velocity",
    "detect_gesture_type",
    "optimize_gesture_timing",
    "validate_coordinates",
    "calculate_safe_gesture_area",
    "estimate_gesture_success_probability",
    "create_fallback_gesture"
]
