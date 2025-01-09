import asyncio
import os

import websockets

from stt.processing import MODEL
from stt.processing.streaming import wssDecode


async def _fun_wrapper(ws):
    """Wrap wssDecode function to add STT Model reference"""
    return await wssDecode(ws, MODEL)


async def WSServer(port: int):
    """Launch the websocket server"""
    async with websockets.serve(_fun_wrapper, "0.0.0.0", serving_port, ping_interval=None, ping_timeout=None):
        await asyncio.Future()


if __name__ == "__main__":
    serving_port = os.environ.get("STREAMING_PORT", 80)
    asyncio.run(WSServer(serving_port))
