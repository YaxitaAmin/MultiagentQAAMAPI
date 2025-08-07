"""Logging utilities for Multi-Agent QA System"""

import sys
from loguru import logger


def setup_logging(level: str = "INFO", log_file: str = "multiagent_qa.log"):
    """Setup logging configuration"""
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file logger
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level
    )
    
    logger.info(f"Logging setup complete - Level: {level}, File: {log_file}")
