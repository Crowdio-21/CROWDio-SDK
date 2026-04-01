# CROWDio SDK

CROWDio SDK provides client-side tools for submitting distributed Python workloads to a CROWDio foreman.

## Included Packages

- `crowdio`: public client APIs, decorators, and image utility helpers.
- `common`: shared protocol and serialization utilities used by the SDK runtime.

## Install

```bash
pip install crowdio-sdk
```

Install optional image processing dependencies:

```bash
pip install "crowdio-sdk[image]"
```

## Quick Start

```python
import asyncio
from crowdio import crowdio_connect, crowdio_map, crowdio_disconnect


def square(x):
    return x * x


async def main():
    await crowdio_connect("localhost", 9000)
    results = await crowdio_map(square, [1, 2, 3, 4])
    print(results)
    await crowdio_disconnect()


asyncio.run(main())
```

## Notes

- The package requires a reachable CROWDio foreman endpoint.
- Image utilities under `crowdio.image_utils` require Pillow (install with `[image]` extra).
