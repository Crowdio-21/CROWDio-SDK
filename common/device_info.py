"""
Device Information Collector
Collects hardware and system specs from PC and Android workers
"""

import platform
import sys
from typing import Dict, Any, Optional


def get_device_specs() -> Dict[str, Any]:
    """
    Collect comprehensive device specifications.
    Works on PC (Windows/Linux/macOS) and Android workers.

    Returns:
        Dictionary with device specifications
    """
    specs = {
        "device_type": "Unknown",
        "os_type": platform.system(),
        "os_version": platform.release(),
        "cpu_model": None,
        "cpu_cores": None,
        "cpu_threads": None,
        "cpu_frequency_mhz": None,
        "ram_total_mb": None,
        "ram_available_mb": None,
        "gpu_model": None,
        "battery_level": None,
        "is_charging": None,
        "network_type": None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    # Detect device type
    if platform.system() == "Linux" and "ANDROID_ROOT" in platform.platform():
        specs["device_type"] = "Android"
    else:
        specs["device_type"] = "PC"

    # Try to get CPU information
    try:
        import psutil

        # CPU details
        specs["cpu_cores"] = psutil.cpu_count(logical=False)
        specs["cpu_threads"] = psutil.cpu_count(logical=True)

        # CPU frequency
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            specs["cpu_frequency_mhz"] = cpu_freq.current

        # RAM information
        mem = psutil.virtual_memory()
        specs["ram_total_mb"] = round(mem.total / (1024 * 1024), 2)
        specs["ram_available_mb"] = round(mem.available / (1024 * 1024), 2)

        # Battery information (for laptops and Android)
        try:
            battery = psutil.sensors_battery()
            if battery:
                specs["battery_level"] = round(battery.percent, 2)
                specs["is_charging"] = battery.power_plugged
        except (AttributeError, RuntimeError):
            # Not available on all systems
            pass

        # Network information
        try:
            net_if_stats = psutil.net_if_stats()
            # Determine primary network type
            active_interfaces = [
                iface for iface, stats in net_if_stats.items() if stats.isup
            ]
            if active_interfaces:
                if any(
                    "wifi" in iface.lower() or "wlan" in iface.lower()
                    for iface in active_interfaces
                ):
                    specs["network_type"] = "WiFi"
                elif any(
                    "eth" in iface.lower() or "ethernet" in iface.lower()
                    for iface in active_interfaces
                ):
                    specs["network_type"] = "Ethernet"
                else:
                    specs["network_type"] = "Unknown"
        except:
            pass

    except ImportError:
        # psutil not available - use basic platform info
        pass

    # Try to get CPU model name
    try:
        import cpuinfo

        cpu_info = cpuinfo.get_cpu_info()
        specs["cpu_model"] = cpu_info.get("brand_raw", None)
    except:
        # cpuinfo not available - try platform-specific methods
        try:
            if platform.system() == "Windows":
                import subprocess

                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        specs["cpu_model"] = lines[1].strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            specs["cpu_model"] = line.split(":")[1].strip()
                            break
        except:
            specs["cpu_model"] = platform.processor() or "Unknown"

    # Try to get GPU information
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            specs["gpu_model"] = gpus[0].name
    except:
        # GPU info not available
        pass

    # Android-specific information
    if specs["device_type"] == "Android":
        try:
            # Try to get Android-specific info using subprocess
            import subprocess

            # Get Android version
            try:
                result = subprocess.run(
                    ["getprop", "ro.build.version.release"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    specs["os_version"] = result.stdout.strip()
            except:
                pass

            # Get device model
            try:
                result = subprocess.run(
                    ["getprop", "ro.product.model"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    device_model = result.stdout.strip()
                    specs["cpu_model"] = f"Android Device: {device_model}"
            except:
                pass

        except:
            pass

    return specs


def get_lightweight_device_specs() -> Dict[str, Any]:
    """
    Get minimal device specs without heavy dependencies.
    Useful for Android workers with limited libraries.

    Returns:
        Dictionary with basic device specifications
    """
    return {
        "device_type": "Android" if ("ANDROID_ROOT" in platform.platform()) else "PC",
        "os_type": platform.system(),
        "os_version": platform.release(),
        "cpu_model": platform.processor() or "Unknown",
        "cpu_cores": None,  # Would need psutil
        "cpu_threads": None,
        "cpu_frequency_mhz": None,
        "ram_total_mb": None,
        "ram_available_mb": None,
        "gpu_model": None,
        "battery_level": None,
        "is_charging": None,
        "network_type": None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def format_device_specs_summary(specs: Dict[str, Any]) -> str:
    """
    Format device specs into a human-readable summary.

    Args:
        specs: Device specifications dictionary

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append(f"Device Type: {specs.get('device_type', 'Unknown')}")
    lines.append(f"OS: {specs.get('os_type', 'Unknown')} {specs.get('os_version', '')}")

    if specs.get("cpu_model"):
        lines.append(f"CPU: {specs['cpu_model']}")

    if specs.get("cpu_cores"):
        cores_info = f"{specs['cpu_cores']} cores"
        if specs.get("cpu_threads"):
            cores_info += f" / {specs['cpu_threads']} threads"
        if specs.get("cpu_frequency_mhz"):
            cores_info += f" @ {specs['cpu_frequency_mhz']:.0f} MHz"
        lines.append(f"CPU: {cores_info}")

    if specs.get("ram_total_mb"):
        ram_info = f"RAM: {specs['ram_total_mb']:.0f} MB total"
        if specs.get("ram_available_mb"):
            ram_info += f" ({specs['ram_available_mb']:.0f} MB available)"
        lines.append(ram_info)

    if specs.get("gpu_model"):
        lines.append(f"GPU: {specs['gpu_model']}")

    if specs.get("battery_level") is not None:
        battery_info = f"Battery: {specs['battery_level']:.0f}%"
        if specs.get("is_charging"):
            battery_info += " (Charging)"
        lines.append(battery_info)

    if specs.get("network_type"):
        lines.append(f"Network: {specs['network_type']}")

    lines.append(f"Python: {specs.get('python_version', 'Unknown')}")

    return "\n".join(lines)


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get real-time performance metrics for MCDM scheduling.

    These metrics change frequently and should be sent periodically.

    Returns:
        Dictionary with current performance metrics
    """
    metrics = {
        "cpu_usage_percent": 0.0,
        "ram_available_mb": 0.0,
        "battery_level": None,
        "is_charging": None,
        "network_speed_mbps": 0.0,
        "storage_available_gb": 0.0,
        "timestamp": None,
    }

    try:
        import psutil
        from datetime import datetime

        # Current CPU usage (average over last interval)
        metrics["cpu_usage_percent"] = psutil.cpu_percent(interval=0.1)

        # Available RAM
        mem = psutil.virtual_memory()
        metrics["ram_available_mb"] = round(mem.available / (1024 * 1024), 2)

        # Battery status (if available)
        try:
            battery = psutil.sensors_battery()
            if battery:
                metrics["battery_level"] = round(battery.percent, 2)
                metrics["is_charging"] = battery.power_plugged
        except (AttributeError, RuntimeError):
            pass

        # Disk space available
        try:
            disk = psutil.disk_usage("/")
            metrics["storage_available_gb"] = round(disk.free / (1024**3), 2)
        except:
            pass

        # Network speed estimation (basic - bytes sent/received in last second)
        try:
            # Get initial counters
            net_io_start = psutil.net_io_counters()
            import time

            time.sleep(0.1)  # Brief sampling
            net_io_end = psutil.net_io_counters()

            # Calculate bytes per second
            bytes_sent = (
                net_io_end.bytes_sent - net_io_start.bytes_sent
            ) * 10  # * 10 for 0.1s sample
            bytes_recv = (net_io_end.bytes_recv - net_io_start.bytes_recv) * 10

            # Convert to Mbps (rough estimate of available bandwidth)
            total_mbps = ((bytes_sent + bytes_recv) * 8) / (1024 * 1024)
            metrics["network_speed_mbps"] = round(total_mbps, 2)
        except:
            pass

        metrics["timestamp"] = datetime.now().isoformat()

    except ImportError:
        # psutil not available
        pass

    return metrics
