# BRL POMDP Interface

This is the ROS-free web interface for `02_BRL_CODE`. Its visual design comes
from the 03 Tomato frontend, but it does not connect to ROS or WebRTC and does
not receive camera images.

## Communication

The browser connects directly to the WebSocket server embedded in `main.py`.

```text
POMDP Planner → InterfaceServer → WebSocket → Frontend Yes/No dialog
POMDP Planner ← InterfaceServer ← WebSocket ← User answer
```

The interface is enabled only when both options are supplied:

```bash
python3 main.py --answer_type human --use-interface
```

Using `--use-interface` with any answer source other than `human` is rejected.
Without `--use-interface`, the existing terminal or automatic-answer behavior
is unchanged.

## Development Server

Node.js 18.12 or newer is required. Node 20 LTS is recommended.

```bash
cd scripts/frontend
npm install
npm run dev
```

Open `http://localhost:9000`. The WebSocket endpoint defaults to
`ws://localhost:9765`.

## Human-query Protocol

Server to frontend:

```json
{
  "type": "ask_fact",
  "value": {
    "request_id": "unique-request-id",
    "fact": "ripe(tomato1)",
    "question": "Is ripe(tomato1) true?",
    "timeout_sec": 300
  }
}
```

Frontend to server:

```json
{
  "command": "query_fact_response",
  "param": {
    "request_id": "unique-request-id",
    "answer": true
  }
}
```

`request_id` prevents a delayed answer from being applied to another question.

## Optional Interface Settings

```text
--interface-host       WebSocket bind address (default: 0.0.0.0)
--interface-port       WebSocket port (default: 9765)
--interface-timeout    Human-answer timeout in seconds (default: 300)
```
