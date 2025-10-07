"""Test that app logging is configured correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the app - this will trigger logging configuration
from app import logger, settings

def test_logging():
    """Test that logging works correctly."""
    print("\n" + "="*80)
    print("Testing App Logging Configuration")
    print("="*80)

    # Log some test messages
    logger.info("Test INFO message")
    logger.debug("Test DEBUG message")
    logger.warning("Test WARNING message")
    logger.error("Test ERROR message")

    # Check that log file exists
    log_file = "logs/agent_api.log"
    if os.path.exists(log_file):
        print(f"\n✅ Log file exists: {log_file}")

        # Get file size
        size = os.path.getsize(log_file)
        print(f"✅ Log file size: {size} bytes")

        if size > 0:
            print("✅ Log file has content")

            # Show last few lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"\n📝 Last 5 log entries:")
                for line in lines[-5:]:
                    print(f"   {line.rstrip()}")
        else:
            print("⚠️  Log file is empty")
    else:
        print(f"❌ Log file not found: {log_file}")

    print("\n" + "="*80)
    print("Logging Configuration:")
    print("="*80)
    print(f"App Name: {settings.app_name}")
    print(f"App Version: {settings.app_version}")
    print(f"Log Directory: {os.path.abspath('logs')}")
    print(f"Log File: {os.path.abspath(log_file)}")
    print("="*80)


if __name__ == "__main__":
    test_logging()
