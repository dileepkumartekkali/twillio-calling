"""FastAPI entrypoint for Render. Exposes:
  - GET  /health -> Render health check
  - /voice (GET+POST) -> Twilio's "A call comes in" webhook; returns TwiML pointing at /ws
  - WS   /ws     -> wss://<your-render-host>/ws, opened by Twilio per the TwiML above
"""
import os

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response

from agent import bot
from pipecat.runner.types import WebSocketRunnerArguments

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request):
    stream_url = f"wss://{request.url.hostname}/ws"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{stream_url}" /></Connect>'
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await bot(WebSocketRunnerArguments(websocket=websocket))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
