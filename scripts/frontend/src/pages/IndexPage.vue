<template>
  <q-page class="flex flex-center">
    <q-btn
      size="xl"
      color="primary"
      label="로봇 호출하기"
      @click="$router.push('/main/current'); sendCommand(ws, 'start_robot')"
    ></q-btn>
  </q-page>
</template>

<script setup>
import { onUnmounted } from 'vue'
import { startSocket, sendCommand } from '../utils/websocket'

const ws = startSocket()


onUnmounted(() => {
  ws.close()
})

</script>