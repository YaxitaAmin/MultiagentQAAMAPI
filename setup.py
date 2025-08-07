"""
Setup script for AMAPI system
Production-ready installation and configuration
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List


def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        "logs",
        "metrics", 
        "screenshots",
        "config",
        "data",
        "cache",
        "temp",
        "exports",
        "backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_requirements():
    """Install required packages"""
    requirements = [
        "asyncio",
        "loguru>=0.7.0",
        "numpy>=1.21.0",
        "anthropic>=0.3.0",
        "openai>=1.0.0",
        "pyautogui>=0.9.54",
        "Pillow>=9.0.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    print("📦 Installing required packages...")
    
    for requirement in requirements:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", requirement
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Installed: {requirement}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {requirement}")


def create_environment_file():
    """Create .env file for environment variables"""
    env_content = """# AMAPI Environment Configuration
# Copy this file to .env and fill in your actual values

# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# System Configuration
AMAPI_ENVIRONMENT=production
AMAPI_LOG_LEVEL=INFO
AMAPI_DEBUG=false

# Android Environment (if using AndroidWorld)
ANDROID_SDK_ROOT=/path/to/android/sdk
ANDROID_HOME=/path/to/android/sdk
ANDROID_AVD_HOME=/path/to/avd

# Optional: Database Configuration
DATABASE_URL=sqlite:///amapi.db

# Optional: Monitoring Configuration
MONITORING_ENABLED=true
METRICS_EXPORT_ENABLED=true

# Optional: Security Configuration
ENCRYPTION_KEY=your_encryption_key_here
"""
    
    env_file = Path(".env.example")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print("✅ Created .env.example file")
    print("📝 Please copy .env.example to .env and configure your API keys")


def validate_configuration():
    """Validate system configuration"""
    config_file = Path("config/system_config.json")
    
    if not config_file.exists():
        print("❌ System configuration file not found")
        return False
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Validate required sections
        required_sections = [
            "system", "logging", "attention", "behavioral", 
            "device_abstraction", "llm", "agents"
        ]
        
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print("✅ Configuration file validated")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Configuration file JSON error: {e}")
        return False


def setup_logging():
    """Setup logging configuration"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file placeholders
    log_files = [
        "amapi_system.log",
        "attention_economics.log", 
        "behavioral_learning.log",
        "device_abstraction.log",
        "agents.log",
        "metrics.log",
        "errors.log"
    ]
    
    for log_file in log_files:
        log_path = logs_dir / log_file
        log_path.touch(exist_ok=True)
    
    print("✅ Logging system configured")


def check_optional_dependencies():
    """Check for optional dependencies"""
    optional_deps = {
        "AndroidWorld": "android_env",
        "Agent-S": "gui_agents", 
        "Selenium": "selenium",
        "OpenCV": "cv2"
    }
    
    print("\n🔍 Checking optional dependencies:")
    
    for name, module in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"⚠️  {name} not available (optional)")


def create_startup_scripts():
    """Create startup scripts"""
    
    # Linux/Mac startup script
    startup_script_unix = """#!/bin/bash
# AMAPI System Startup Script

echo "🚀 Starting AMAPI System..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check environment variables
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure."
    exit 1
fi

# Start AMAPI system
python main.py --config config/system_config.json

echo "✅ AMAPI System startup complete"
"""
    
    # Windows startup script
    startup_script_windows = """@echo off
REM AMAPI System Startup Script

echo 🚀 Starting AMAPI System...

REM Check if virtual environment exists
if exist "venv\\Scripts\\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Check environment variables
if not exist ".env" (
    echo ❌ .env file not found. Please copy .env.example to .env and configure.
    exit /b 1
)

REM Start AMAPI system
python main.py --config config/system_config.json

echo ✅ AMAPI System startup complete
pause
"""
    
    # Write scripts
    with open("start_amapi.sh", "w") as f:
        f.write(startup_script_unix)
    os.chmod("start_amapi.sh", 0o755)
    
    with open("start_amapi.bat", "w") as f:
        f.write(startup_script_windows)
    
    print("✅ Created startup scripts")


def create_readme():
    """Create comprehensive README"""
    readme_content = """# AMAPI - Advanced Multi-Agent Performance Intelligence

## Overview

AMAPI is a production-ready multi-agent system that combines attention economics, behavioral learning, and device abstraction for intelligent task automation and QA.

## Features

- 🧠 **Attention Economics**: Intelligent attention allocation and management
- 📚 **Behavioral Learning**: Adaptive pattern recognition and learning
- 📱 **Device Abstraction**: Universal device compatibility and adaptation
- 🤖 **Multi-Agent System**: Supervisor, Planner, Executor, and Verifier agents
- 🔧 **AndroidWorld Integration**: Advanced Android automation capabilities
- 📊 **Comprehensive Monitoring**: Real-time metrics and system evaluation

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run Setup**
   ```bash
   python setup.py
   ```

4. **Start System**
   ```bash
   # Linux/Mac
   ./start_amapi.sh
   
   # Windows
   start_amapi.bat
   
   # Or directly
   python main.py
   ```

## Configuration

System configuration is in `config/system_config.json`. Key sections:

- **System**: Core system settings
- **Attention**: Attention economics parameters
- **Behavioral**: Learning and adaptation settings
- **Agents**: Individual agent configurations
- **LLM**: Language model integration
- **Android**: AndroidWorld integration settings

## Usage Examples

### Interactive Mode
```bash
python main.py --interactive
```

### Single Task Execution
```bash
python main.py --task "Open settings and navigate to WiFi"
```

### Daemon Mode
```bash
python main.py --daemon
```

## Architecture

```
AMAPI System
├── Core Components
│   ├── Attention Economics Engine
│   ├── Behavioral Learning Engine
│   ├── Device Abstraction Layer
│   └── LLM Interface
├── Agents
│   ├── Supervisor Agent
│   ├── Planner Agent
│   ├── Executor Agent
│   └── Verifier Agent
├── Integrations
│   ├── AndroidWorld Environment
│   └── Agent-S Framework
└── Monitoring
    ├── System Metrics
    ├── Performance Evaluator
    └── Health Monitor
```

## API Keys Setup

1. **Anthropic Claude API**
   - Get API key from: https://console.anthropic.com/
   - Add to `.env`: `ANTHROPIC_API_KEY=your_key_here`

2. **OpenAI API** (optional fallback)
   - Get API key from: https://platform.openai.com/
   - Add to `.env`: `OPENAI_API_KEY=your_key_here`

## Android Environment Setup (Optional)

For AndroidWorld integration:

1. Install Android SDK
2. Create Android AVD (Android Virtual Device)
3. Configure paths in `.env`:
   ```
   ANDROID_SDK_ROOT=/path/to/android/sdk
   ANDROID_AVD_HOME=/path/to/avd
   ```

## Monitoring and Metrics

- System metrics: `http://localhost:8080/metrics` (if web interface enabled)
- Log files: `logs/` directory
- Exported metrics: `metrics/` directory
- Screenshots: `screenshots/` directory

## Production Deployment

1. **Environment Setup**
   - Use production configuration
   - Set up proper logging
   - Configure monitoring
   - Set up backup systems

2. **Security**
   - Encrypt API keys
   - Use secure communication
   - Enable audit logging
   - Set up access control

3. **Scaling**
   - Configure resource limits
   - Set up load balancing
   - Enable caching
   - Monitor performance

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

2. **API Key Issues**
   - Verify API keys in `.env` file
   - Check API key permissions
   - Verify network connectivity

3. **Android Integration Issues**
   - Ensure Android SDK is installed
   - Check AVD configuration
   - Verify ADB is in PATH

### Debug Mode

Enable debug mode in configuration:
```json
{
  "system": {
    "debug_mode": true
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

Copyright (c) 2024 AMAPI Project. All rights reserved.

## Support

For support and questions:
- Check documentation
- Review configuration
- Enable debug logging
- Check system health
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created comprehensive README.md")


def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """# AMAPI Core Requirements
asyncio
loguru>=0.7.0
numpy>=1.21.0
anthropic>=0.3.0
openai>=1.0.0
pyautogui>=0.9.54
Pillow>=9.0.0
requests>=2.28.0
aiohttp>=3.8.0
pydantic>=2.0.0
python-dotenv>=1.0.0

# Optional: AndroidWorld Integration
# android-env>=1.0.0

# Optional: Agent-S Integration
# gui-agents>=1.0.0

# Optional: Advanced Computer Vision
# opencv-python>=4.7.0

# Optional: Web Automation
# selenium>=4.0.0

# Optional: Database Support
# sqlalchemy>=2.0.0
# alembic>=1.11.0

# Development Dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
mypy>=1.4.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")


def run_system_checks():
    """Run comprehensive system checks"""
    print("\n🔍 Running system checks...")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Python version
    total_checks += 1
    if sys.version_info >= (3, 8):
        print("✅ Python version check passed")
        checks_passed += 1
    else:
        print("❌ Python version check failed")
    
    # Check 2: Configuration file
    total_checks += 1
    if validate_configuration():
        checks_passed += 1
    
    # Check 3: Directory structure
    total_checks += 1
    required_dirs = ["logs", "config", "screenshots", "metrics"]
    if all(Path(d).exists() for d in required_dirs):
        print("✅ Directory structure check passed")
        checks_passed += 1
    else:
        print("❌ Directory structure check failed")
    
    # Check 4: Environment file
    total_checks += 1
    if Path(".env.example").exists():
        print("✅ Environment template check passed")
        checks_passed += 1
    else:
        print("❌ Environment template check failed")
    
    print(f"\n📊 System checks: {checks_passed}/{total_checks} passed")
    
    if checks_passed == total_checks:
        print("🎉 All system checks passed! AMAPI is ready to run.")
        return True
    else:
        print("⚠️  Some system checks failed. Please review and fix issues.")
        return False


def main():
    """Main setup function"""
    print("🚀 AMAPI System Setup")
    print("=" * 50)
    
    # Run setup steps
    check_python_version()
    create_directory_structure()
    create_requirements_file()
    install_requirements()
    create_environment_file()
    setup_logging()
    create_startup_scripts()
    create_readme()
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Run final system checks
    if run_system_checks():
        print("\n✅ AMAPI setup completed successfully!")
        print("\n📝 Next steps:")
        print("1. Copy .env.example to .env")
        print("2. Configure your API keys in .env")
        print("3. Run: python main.py --interactive")
        print("4. Or use startup scripts: ./start_amapi.sh")
    else:
        print("\n❌ Setup completed with issues. Please review and fix.")
        sys.exit(1)


if __name__ == "__main__":
    main()