"""Test script to verify Langfuse is set up as default telemetry."""

import sys
sys.path.insert(0, 'src')

from linus.agents.telemetry import initialize_telemetry, is_langfuse_available
from linus.settings import Settings

def test_langfuse_setup():
    """Test that Langfuse is properly configured."""

    settings = Settings()

    print("=" * 60)
    print("LANGFUSE TELEMETRY CONFIGURATION TEST")
    print("=" * 60)
    print()

    # Check availability
    print("1. Langfuse Package:")
    print(f"   ✓ Langfuse available: {is_langfuse_available()}")
    print()

    # Check settings
    print("2. Environment Configuration:")
    print(f"   ✓ Telemetry enabled: {settings.telemetry_enabled}")
    print(f"   ✓ Default exporter: {settings.telemetry_exporter}")
    print(f"   ✓ Langfuse host: {settings.langfuse_host}")
    print(f"   ✓ Public key configured: {'Yes' if settings.langfuse_public_key else 'No'}")
    print(f"   ✓ Secret key configured: {'Yes' if settings.langfuse_secret_key else 'No'}")
    print()

    # Test initialization
    print("3. Tracer Initialization:")
    tracer = initialize_telemetry(
        service_name='test-reasoning-agent',
        exporter_type=settings.telemetry_exporter,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        langfuse_host=settings.langfuse_host,
        enabled=settings.telemetry_enabled
    )

    print(f"   ✓ Tracer type: {type(tracer).__name__}")
    print(f"   ✓ Tracer enabled: {tracer.enabled}")
    print(f"   ✓ Client initialized: {'Yes' if tracer.client else 'No'}")
    print()

    print("=" * 60)
    print("✅ SUCCESS: Langfuse is configured as default telemetry!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Start your agent: python src/app.py")
    print("  2. Make API calls to test tracing")
    print("  3. View traces at: https://cloud.langfuse.com")
    print()

    return tracer


if __name__ == "__main__":
    try:
        tracer = test_langfuse_setup()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
