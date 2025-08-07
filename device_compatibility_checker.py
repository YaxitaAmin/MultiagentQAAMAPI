"""
Device Compatibility Checker - Missing component for AMAPI system
Provides device compatibility analysis and validation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Device compatibility levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INCOMPATIBLE = "incompatible"


@dataclass
class DeviceProfile:
    """Device profile for compatibility checking"""
    device_type: str
    screen_width: int
    screen_height: int
    density: int
    api_level: int
    platform: str = "android"
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class CompatibilityResult:
    """Result of compatibility check"""
    level: CompatibilityLevel
    score: float  # 0.0 to 1.0
    issues: List[str]
    recommendations: List[str]
    supported_features: List[str]


class DeviceCompatibilityChecker:
    """
    Device compatibility checker for AMAPI system
    Analyzes device capabilities and compatibility
    """
    
    def __init__(self, reference_profile: Optional[DeviceProfile] = None):
        """
        Initialize compatibility checker
        
        Args:
            reference_profile: Reference device profile for comparison
        """
        self.reference_profile = reference_profile or DeviceProfile(
            device_type="emulator",
            screen_width=1080,
            screen_height=1920,
            density=420,
            api_level=29,
            platform="android",
            capabilities=["touch", "gestures", "screenshots", "ui_automation"]
        )
        
        # Minimum requirements
        self.min_requirements = {
            'screen_width': 720,
            'screen_height': 1280,
            'density': 240,
            'api_level': 21
        }
        
        # Optimal requirements
        self.optimal_requirements = {
            'screen_width': 1080,
            'screen_height': 1920,
            'density': 420,
            'api_level': 28
        }
        
        logger.info("Device Compatibility Checker initialized")
    
    def check_compatibility(self, device_profile: DeviceProfile) -> CompatibilityResult:
        """
        Check device compatibility
        
        Args:
            device_profile: Device profile to check
            
        Returns:
            Compatibility result with level, score, and recommendations
        """
        try:
            issues = []
            recommendations = []
            supported_features = []
            
            # Check minimum requirements
            compatibility_score = 1.0
            
            # Screen resolution check
            if device_profile.screen_width < self.min_requirements['screen_width']:
                issues.append(f"Screen width {device_profile.screen_width} below minimum {self.min_requirements['screen_width']}")
                compatibility_score -= 0.2
                recommendations.append("Use device with higher screen resolution")
            
            if device_profile.screen_height < self.min_requirements['screen_height']:
                issues.append(f"Screen height {device_profile.screen_height} below minimum {self.min_requirements['screen_height']}")
                compatibility_score -= 0.2
                recommendations.append("Use device with higher screen resolution")
            
            # Density check
            if device_profile.density < self.min_requirements['density']:
                issues.append(f"Screen density {device_profile.density} below minimum {self.min_requirements['density']}")
                compatibility_score -= 0.1
                recommendations.append("Use device with higher screen density")
            
            # API level check
            if device_profile.api_level < self.min_requirements['api_level']:
                issues.append(f"API level {device_profile.api_level} below minimum {self.min_requirements['api_level']}")
                compatibility_score -= 0.3
                recommendations.append("Upgrade to newer Android version")
            
            # Platform check
            if device_profile.platform.lower() != "android":
                issues.append(f"Platform {device_profile.platform} not fully supported")
                compatibility_score -= 0.2
                recommendations.append("Use Android device for optimal compatibility")
            
            # Check capabilities
            required_capabilities = ["touch", "ui_automation"]
            for capability in required_capabilities:
                if capability in device_profile.capabilities:
                    supported_features.append(capability)
                else:
                    issues.append(f"Missing required capability: {capability}")
                    compatibility_score -= 0.1
            
            # Optional capabilities
            optional_capabilities = ["gestures", "screenshots", "multitouch"]
            for capability in optional_capabilities:
                if capability in device_profile.capabilities:
                    supported_features.append(capability)
                else:
                    compatibility_score -= 0.05
            
            # Determine compatibility level
            compatibility_score = max(0.0, compatibility_score)
            
            if compatibility_score >= 0.9:
                level = CompatibilityLevel.EXCELLENT
            elif compatibility_score >= 0.7:
                level = CompatibilityLevel.GOOD
            elif compatibility_score >= 0.5:
                level = CompatibilityLevel.FAIR
            elif compatibility_score >= 0.3:
                level = CompatibilityLevel.POOR
            else:
                level = CompatibilityLevel.INCOMPATIBLE
            
            # Add positive recommendations
            if compatibility_score >= 0.8:
                recommendations.append("Device well-suited for AMAPI system")
            
            if not recommendations:
                recommendations.append("Device meets basic requirements")
            
            result = CompatibilityResult(
                level=level,
                score=compatibility_score,
                issues=issues,
                recommendations=recommendations,
                supported_features=supported_features
            )
            
            logger.info(f"Compatibility check completed: {level.value} (score: {compatibility_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return CompatibilityResult(
                level=CompatibilityLevel.INCOMPATIBLE,
                score=0.0,
                issues=[f"Compatibility check failed: {e}"],
                recommendations=["Contact support for assistance"],
                supported_features=[]
            )
    
    def is_compatible(self, device_profile: DeviceProfile, min_level: CompatibilityLevel = CompatibilityLevel.FAIR) -> bool:
        """
        Check if device meets minimum compatibility level
        
        Args:
            device_profile: Device profile to check
            min_level: Minimum required compatibility level
            
        Returns:
            True if device is compatible
        """
        result = self.check_compatibility(device_profile)
        
        level_order = {
            CompatibilityLevel.INCOMPATIBLE: 0,
            CompatibilityLevel.POOR: 1,
            CompatibilityLevel.FAIR: 2,
            CompatibilityLevel.GOOD: 3,
            CompatibilityLevel.EXCELLENT: 4
        }
        
        return level_order[result.level] >= level_order[min_level]
    
    def get_optimization_recommendations(self, device_profile: DeviceProfile) -> List[str]:
        """
        Get device-specific optimization recommendations
        
        Args:
            device_profile: Device profile
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        try:
            # Resolution-based recommendations
            if device_profile.screen_width < self.optimal_requirements['screen_width']:
                recommendations.append("Consider using higher resolution device for better UI element detection")
            
            # Density-based recommendations
            if device_profile.density < self.optimal_requirements['density']:
                recommendations.append("Higher density display will improve touch target accuracy")
            
            # API level recommendations
            if device_profile.api_level < self.optimal_requirements['api_level']:
                recommendations.append("Newer Android version provides better automation APIs")
            
            # Device type recommendations
            if device_profile.device_type == "physical":
                recommendations.append("Physical devices may have variable performance")
            elif device_profile.device_type == "emulator":
                recommendations.append("Emulators provide consistent testing environment")
            
            # Capability recommendations
            if "multitouch" not in device_profile.capabilities:
                recommendations.append("Enable multitouch for advanced gesture testing")
            
            if "screenshots" not in device_profile.capabilities:
                recommendations.append("Enable screenshot capability for visual verification")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Contact support for device optimization guidance"]
    
    def create_device_profile(self, device_info: Dict[str, Any]) -> DeviceProfile:
        """
        Create device profile from device information
        
        Args:
            device_info: Dictionary containing device information
            
        Returns:
            DeviceProfile object
        """
        try:
            return DeviceProfile(
                device_type=device_info.get('device_type', 'unknown'),
                screen_width=device_info.get('screen_width', 1080),
                screen_height=device_info.get('screen_height', 1920),
                density=device_info.get('density', 420),
                api_level=device_info.get('api_level', 29),
                platform=device_info.get('platform', 'android'),
                capabilities=device_info.get('capabilities', ['touch', 'ui_automation'])
            )
        except Exception as e:
            logger.error(f"Failed to create device profile: {e}")
            return self.reference_profile
    
    def compare_devices(self, device1: DeviceProfile, device2: DeviceProfile) -> Dict[str, Any]:
        """
        Compare two device profiles
        
        Args:
            device1: First device profile
            device2: Second device profile
            
        Returns:
            Comparison result
        """
        try:
            result1 = self.check_compatibility(device1)
            result2 = self.check_compatibility(device2)
            
            return {
                'device1_compatibility': result1,
                'device2_compatibility': result2,
                'better_device': 'device1' if result1.score > result2.score else 'device2',
                'score_difference': abs(result1.score - result2.score),
                'comparison_summary': self._generate_comparison_summary(device1, device2, result1, result2)
            }
            
        except Exception as e:
            logger.error(f"Device comparison failed: {e}")
            return {'error': str(e)}
    
    def _generate_comparison_summary(self, device1: DeviceProfile, device2: DeviceProfile, 
                                   result1: CompatibilityResult, result2: CompatibilityResult) -> str:
        """Generate comparison summary"""
        try:
            if result1.score > result2.score:
                return f"Device 1 is more compatible ({result1.level.value} vs {result2.level.value})"
            elif result2.score > result1.score:
                return f"Device 2 is more compatible ({result2.level.value} vs {result1.level.value})"
            else:
                return f"Both devices have similar compatibility ({result1.level.value})"
        except:
            return "Comparison summary unavailable"


# Export classes for import
__all__ = [
    'DeviceCompatibilityChecker',
    'DeviceProfile', 
    'CompatibilityResult',
    'CompatibilityLevel'
]