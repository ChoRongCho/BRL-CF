"""ROS-free WebSocket bridge between the POMDP runner and the web UI."""

import asyncio
import json
import threading
import uuid

import websockets


class InterfaceServer:
    """Run a WebSocket server in the background and provide blocking queries."""

    def __init__(self, host="0.0.0.0", port=9765, answer_timeout=300.0):
        self.host = host
        self.port = int(port)
        self.answer_timeout = float(answer_timeout)
        self._loop = None
        self._server = None
        self._thread = None
        self._started = threading.Event()
        self._clients_available = threading.Event()
        self._clients = set()
        self._pending_lock = threading.Lock()
        self._pending = {}
        self._startup_error = None

    def start(self):
        """Start the WebSocket event loop and wait until its socket is ready."""
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            name="pomdp-interface-server",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise RuntimeError("Timed out while starting the interface server")
        if self._startup_error is not None:
            raise RuntimeError(
                "Failed to start the interface server: %s" % self._startup_error
            )

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_server())
        except Exception as error:
            self._startup_error = error
            self._started.set()
            return

        self._started.set()
        self._loop.run_forever()

    async def _start_server(self):
        self._server = await websockets.serve(self._handler, self.host, self.port)

    async def _handler(self, websocket, *unused_args):
        self._clients.add(websocket)
        self._clients_available.set()
        await websocket.send(json.dumps({"type": "connected", "value": {}}))
        try:
            async for raw_message in websocket:
                await self._handle_message(raw_message)
        finally:
            self._clients.discard(websocket)
            if not self._clients:
                self._clients_available.clear()

    async def _handle_message(self, raw_message):
        try:
            message = json.loads(raw_message)
        except (TypeError, ValueError):
            return

        if message.get("command") != "query_fact_response":
            return

        payload = message.get("param") or {}
        request_id = str(payload.get("request_id", ""))
        raw_answer = payload.get("answer")
        if isinstance(raw_answer, bool):
            answer = raw_answer
        elif str(raw_answer).strip().lower() in ("true", "yes", "1", "y"):
            answer = True
        elif str(raw_answer).strip().lower() in ("false", "no", "0", "n"):
            answer = False
        else:
            return

        with self._pending_lock:
            pending = self._pending.get(request_id)
            if pending is None:
                return
            pending["answer"] = answer
            pending["event"].set()

    async def _broadcast(self, message):
        if not self._clients:
            return
        encoded = json.dumps(message, ensure_ascii=False)
        disconnected = []
        for client in tuple(self._clients):
            try:
                await client.send(encoded)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
        for client in disconnected:
            self._clients.discard(client)
        if not self._clients:
            self._clients_available.clear()

    def publish(self, message_type, value):
        """Send a non-blocking status event to every connected frontend."""
        if self._loop is None or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast({"type": message_type, "value": value}), self._loop
        )

    def ask(self, fact, question=None):
        """
        Send one fact question and block the planner thread for a Yes/No answer.

        입력 예: fact="ripe(tomato1)", question="ripe(tomato1) is true?"
        출력 예: 사용자가 Yes를 누르면 True, No를 누르면 False
        """
        if not self._clients_available.wait(timeout=30.0):
            raise RuntimeError(
                "No frontend is connected. Open the web UI before a human query."
            )

        request_id = uuid.uuid4().hex
        pending = {"event": threading.Event(), "answer": None}
        with self._pending_lock:
            self._pending[request_id] = pending

        self.publish(
            "ask_fact",
            {
                "request_id": request_id,
                "fact": str(fact),
                "question": question or ("Is %s true?" % fact),
                "timeout_sec": self.answer_timeout,
            },
        )

        answered = pending["event"].wait(timeout=self.answer_timeout)
        with self._pending_lock:
            self._pending.pop(request_id, None)
        if not answered:
            raise TimeoutError("Human feedback timed out for fact: %s" % fact)
        return pending["answer"]

