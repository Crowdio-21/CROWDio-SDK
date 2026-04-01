#!/usr/bin/env python3
"""
Example client script demonstrating CrowdCompute usage
"""

import sys
import os
import asyncio
import time

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crowdio import crowdio_connect, crowdio_map, crowdio_disconnect


def square(x):
    """Simple function to square a number"""
    import time
    time.sleep(0.1)  # Simulate some work
    return x ** 2


def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def process_data(data):
    """Process some data"""
    import time
    
    # Simulate some computation
    result = sum(data) * 2
    time.sleep(0.05)
    return result


async def main():
    """Main example function"""
    if len(sys.argv) != 2:
        print("Usage: python example_client.py <foreman_host>")
        print("Example: python example_client.py 192.168.1.10")
        sys.exit(1)
    
    foreman_host = sys.argv[1]
    
    print("CrowdCompute Example Client")
    print("=" * 40)
    
    try:
        # Connect to foreman
        print(f"Connecting to foreman at {foreman_host}:9000...")
        await crowdio_connect(foreman_host, 9000)
        print("Connected!")
        
        # Example 1: Simple map operation
        print("\n1. Running square function on numbers 1-10...")
        numbers = list(range(1, 11))
        start_time = time.time()
        results = await crowdio_map(square, numbers)
        end_time = time.time()
        
        print(f"Results: {results}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Example 2: Fibonacci calculation
        print("\n2. Calculating fibonacci numbers...")
        fib_inputs = [10, 15, 20, 25, 30]
        start_time = time.time()
        fib_results = await crowdio_map(fibonacci, fib_inputs)
        end_time = time.time()
        
        print(f"Fibonacci results: {fib_results}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Example 3: Data processing
        print("\n3. Processing data arrays...")
        data_arrays = [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [100, 200, 300, 400, 500],
            [1000, 2000, 3000, 4000, 5000]
        ]
        start_time = time.time()
        processed_results = await crowdio_map(process_data, data_arrays)
        end_time = time.time()
        
        print(f"Processed results: {processed_results}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # # Example 4: Single function execution
        # print("\n4. Running single function...")
        # start_time = time.time()
        # single_result = await run(square, 42)
        # end_time = time.time()
        
        # print(f"Single result: {single_result}")
        # print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect
        await crowdio_disconnect()
        print("Disconnected from foreman")


if __name__ == "__main__":
    asyncio.run(main())
