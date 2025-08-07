"""Configuration Manager for Multi-Agent QA System"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ConfigManager:
    """Manages system configuration"""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration manager"""
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm_provider": "anthropic",
            "llm_model": "claude-3-5-sonnet-20241022",
            "task_name": "settings_wifi",
            "emulator_name": "AndroidWorldAvd",
            "screenshots_dir": "screenshots",
            "attention_budgets": {
                "planner": 120.0,
                "executor": 80.0,
                "verifier": 100.0,
                "supervisor": 150.0
            },
            "max_steps": 30,
            "enable_attention_economics": True,
            "enable_benchmarking": True
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        self.config.update(updates)
        logger.info("Configuration updated")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
