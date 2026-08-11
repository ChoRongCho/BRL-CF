// =============================================================================
// WebSocket 서버 설정
// =============================================================================
// 배포 환경에 맞게 IP 주소를 변경하세요:
// - 개발 환경: ws://localhost:8765
// - 시뮬레이션: ws://192.168.50.127:8765
// - 실제 로봇: ws://192.168.50.129:8765
// 포트 8765는 brl_server/scripts/websocket.py에서 사용
// =============================================================================
const WS_SERVER_IP = window.location.hostname
const WS_SERVER_PORT = '9765'
const wsserver = `ws://${WS_SERVER_IP}:${WS_SERVER_PORT}`

// WebSocket wrapper with auto-reconnect
let wsInstance = null
let reconnectTimer = null
let messageHandler = null

export function startSocket() {
    if (wsInstance && wsInstance.readyState === WebSocket.OPEN) {
        return wsInstance
    }

    wsInstance = new WebSocket(wsserver)

    wsInstance.onopen = () => {
        console.log('[WS] Connected')
        if (reconnectTimer) {
            clearTimeout(reconnectTimer)
            reconnectTimer = null
        }
    }

    wsInstance.onerror = (err) => {
        console.error('[WS] Error:', err)
    }

    wsInstance.onclose = () => {
        console.log('[WS] Disconnected - will reconnect in 3s')
        scheduleReconnect()
    }

    // Re-attach message handler if exists
    if (messageHandler) {
        wsInstance.onmessage = messageHandler
    }

    return wsInstance
}

function scheduleReconnect() {
    if (reconnectTimer) return
    reconnectTimer = setTimeout(() => {
        console.log('[WS] Attempting reconnect...')
        reconnectTimer = null
        startSocket()
    }, 3000)
}

export function setMessageHandler(handler) {
    messageHandler = handler
    if (wsInstance) {
        wsInstance.onmessage = handler
    }
}

export function sendCommand(ws, command, param = null) {
    // Use global instance if ws is invalid
    const socket = (ws && ws.readyState === WebSocket.OPEN) ? ws : wsInstance

    if (!socket || socket.readyState !== WebSocket.OPEN) {
        console.error('[WS] Not connected. Command queued for reconnect:', command)
        // Trigger reconnect
        scheduleReconnect()
        return false
    }

    const message = { command }
    if (param !== null) {
        message.param = param
    }
    console.log('[WS] Sending:', command, param)
    socket.send(JSON.stringify(message))
    return true
}

export function stopSocket(ws) {
    const socket = ws || wsInstance
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close()
        console.log('[WS] Closed')
    } else {
        console.warn('[WS] Already closed')
    }
}

export function getSocket() {
    return wsInstance
}
