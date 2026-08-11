<template>
    <q-page class="row q-mx-lg">
        <div class="col-2">
            <div class="q-pa-md row">
                <div class="col text-center" @click="showQuitRobotDialog = true">
                    <q-icon name="power_settings_new" size="xl"></q-icon>
                    <div>종료</div>
                </div>
                <div class="col text-center">
                    <q-icon name="settings" size="xl"></q-icon>
                    <div>로봇관리</div>
                </div>
            </div>
            <div class="q-pa-md text-center">
                <div class="text-h6 text-bold">
                    작업 정보
                </div>
                <q-card flat bordered class="my-card">
                    <q-card-section class="text-bold text-h6">
                        <div>수확량</div>
                        <div class="text-primary">23개</div>
                    </q-card-section>

                    <q-separator inset />

                    <q-card-section class="text-bold text-h6">
                        <div>소요시간</div>
                        <div class="text-primary">00시간 17분</div>
                    </q-card-section>
                </q-card>
            </div>
            <div class="q-pa-md text-center">
                <div class="text-h6 text-bold">
                    조작 방법
                    <q-icon name="help_outline" size="md">
                        <q-popup-proxy>
                            <div class="q-pa-md text-center ">
                                <div class="text-h6">조작 방법 도우미</div>
                                <div class="q-mt-md" style="white-space: pre;">
                                    1. ‘말하기 버튼‘을 누르시면, <br>
                                    말하기로 조작을 할 수 있습니다. <br><br>
                                    2. ‘손짓하기 버튼‘을 누르시면, <br>
                                    손짓하기로 조작을 할 수 있습니다. <br><br>
                                    3. ‘컨트롤러 사용하기 버튼’을 누르시면, <br>
                                    컨트롤러로 조작을 할 수 있습니다. <br>
                                </div>
                            </div>
                        </q-popup-proxy>
                    </q-icon>
                </div>
                <div class="q-pt-sm q-gutter-y-sm">
                    <q-btn 
                        v-for="mode in ctr_modes"
                        :key="mode.value"
                        class="full-width" 
                        :color="ctr_mode.includes(mode.value) ? 'primary' : 'green-2'" 
                        :text-color="ctr_mode === mode.value ? 'white' : 'black'" 
                        size="lg" 
                        @click="mode.clicked"
                        style="white-space: pre; height: 80px"
                    >
                        <q-icon :name="mode.icon" class="q-mr-lg"></q-icon>
                        <div>{{ mode.label }}</div>
                    </q-btn>
                </div>
            </div>
        </div>
        <div class="col q-pa-md q-px-xl">
            <!-- CF 모드일 때 상태창 -->
            <div v-if="cfStep !== 'idle'">
                <div class="bg-orange-2 text-h6 text-bold q-pa-sm" style="border: 3px solid #FF9800;">
                    <q-chip color="orange" text-color="white">CF 모드</q-chip>
                    <span v-if="cfStep === 'select_type'">수정 유형을 선택하세요</span>
                    <span v-else-if="cfStep === 'select_pixel'">
                        <q-icon name="ads_click" class="q-mr-sm" />
                        화면에서 토마토를 클릭하세요 ({{ cfSelectedType }})
                        <span v-if="cfWarningMsg" class="text-red q-ml-md">
                            <q-icon name="warning" /> {{ cfWarningMsg }}
                        </span>
                    </span>
                    <span v-else-if="cfStep === 'select_status'">
                        상태를 선택하세요 ({{ cfSelectedType }})
                        <span v-if="lastClickedPixel.x !== null"> - 위치: ({{ lastClickedPixel.x }}, {{ lastClickedPixel.y }})</span>
                    </span>
                    <q-btn flat dense color="grey-8" icon="close" class="q-ml-md" @click="cancelCF" />
                </div>
            </div>
            <!-- 일반 상태창 -->
            <div v-else-if="is_playing">
                <div class="bg-white text-h6 text-bold q-pa-sm" style="border: 3px solid #02542D;" >
                <q-chip :color="status.color" text-color="black">{{ status.label }}</q-chip>
                {{ status.msg }}</div>
            </div>
            <div v-else>
                <div class="bg-white text-h6 text-bold q-pa-sm" style="border: 3px solid #02542D;" >
                <q-chip color="red-2" text-color="black">작업중지</q-chip>
                작업을 계속 하려면 [작업 시작]을 눌러주세요!</div>
            </div>

            <div class="q-pt-lg">
                <div style="position: relative;">
                    <video
                        ref="rosCameraMain"
                        autoplay playsinline muted
                        style=" width: 100%; aspect-ratio: 4 / 3; background-color: black;"
                        :style="cfStep === 'select_pixel' ? 'cursor: crosshair; border: 6px solid #FF9800;' : (status.value === 'finding_missed_tomato' ? 'cursor: pointer; border: 6px solid #0000FF': 'border: 3px solid #02542D;')"
                        @click="videoClicked"
                    ></video>
                    <!-- ============================================================
                         CF 픽셀 선택 모드: Workspace 외 영역 어둡게 처리
                         ============================================================
                         로봇이 실제로 닿을 수 있는 영역만 밝게 표시하여
                         사용자가 유효한 토마토만 선택하도록 유도

                         유효 영역 (원본 해상도 640x480 기준):
                           - X: 280 ~ 640 (왼쪽 경계)
                           - Y: 60 ~ 450 (상/하단 경계)

                         퍼센트 계산:
                           - 왼쪽: 280/640 = 43.75%
                           - 상단: 60/480 = 12.5%
                           - 하단: (480-450)/480 = 6.25%

                         ┌─────────────────────────────────┐
                         │▓▓▓▓▓▓▓▓▓▓▓│    어두움 (상단)   │
                         │▓▓▓▓▓▓▓▓▓▓▓├────────────────────┤
                         │▓▓ 어두움  │                    │
                         │▓▓ (왼쪽)  │   밝음 (유효영역)  │
                         │▓▓▓▓▓▓▓▓▓▓▓│                    │
                         │▓▓▓▓▓▓▓▓▓▓▓├────────────────────┤
                         │▓▓▓▓▓▓▓▓▓▓▓│    어두움 (하단)   │
                         └─────────────────────────────────┘
                    -->
                    <!-- 왼쪽 영역 (x < 280) -->
                    <div v-if="cfStep === 'select_pixel'"
                        style="position: absolute; top: 0; left: 0; width: 43.75%; height: 100%;
                               background: rgba(0,0,0,0.6); pointer-events: none; z-index: 5;">
                    </div>
                    <!-- 상단 영역 (y < 60, 왼쪽 제외) -->
                    <div v-if="cfStep === 'select_pixel'"
                        style="position: absolute; top: 0; left: 43.75%; right: 0; height: 12.5%;
                               background: rgba(0,0,0,0.6); pointer-events: none; z-index: 5;">
                    </div>
                    <!-- 하단 영역 (y > 450, 왼쪽 제외) -->
                    <div v-if="cfStep === 'select_pixel'"
                        style="position: absolute; bottom: 0; left: 43.75%; right: 0; height: 6.25%;
                               background: rgba(0,0,0,0.6); pointer-events: none; z-index: 5;">
                    </div>
                    <!-- CF 픽셀 선택 모드에서는 서브카메라 숨김 (v-show로 스트림 유지) -->
                    <div v-show="cfStep !== 'select_pixel'" class="absolute-top-right" style="width: 33%;">
                        <q-chip class="absolute-top-right q-ma-md" color="primary" text-color="white" icon="refresh" clickable=""
                            @click="switchCam" style="z-index: 10;"
                        >화면 바꾸기</q-chip>
                        <video ref="rosCameraSub"
                            autoplay playsinline muted
                            style="border: 3px solid #02542D; width: 100%; aspect-ratio: 4 / 3; background-color: black;"
                        ></video>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-1" style="margin-top: 100px">
            <div
                v-for="btn in robot_btns"
                :key="btn.value"
                class="q-mt-md"
            >
                <q-btn
                    color="secondary"
                    size="lg"
                    text-color="black"
                    :disable="(!status.availabe_btns.includes(btn.value) && btn.value !== 'cf_trigger') || (btn.value === 'cf_trigger' && currentMain !== 'cam1')"
                    @click="clickRobotBtn(btn.value)"
                    v-if="btn.show()"
                >
                    <div class="column text-center">
                        <q-icon center size="4em" :name="btn.icon" :color="(status.availabe_btns.includes(btn.value) || btn.value === 'cf_trigger') && !(btn.value === 'cf_trigger' && currentMain !== 'cam1') ? btn.color : 'grey-6'" />
                        <div>{{ btn.label }}</div>
                    </div>
                </q-btn>
            </div>
            <div class="col-4"></div>
        </div>
        <div class="fixed-bottom-right q-ma-lg q-pa-xl text-center text-white" style="background-color: #000000aa;" v-if="dialog">
            <div class="row q-col-gutter-x-md" v-if="dialog === 'select_correction'">
                <div class="q-mt-xl">
                    <q-icon class="cursor-pointer" style="vertical-align: center;" name="cancel" size="3em" @click="closeDialog"></q-icon>
                </div>
                <div class="text-h5 row q-mt-xl">도움이 필요하신가요?</div>
                <div
                    v-for="btn in correction_btns"
                    :key="btn.value"
                >
                    <q-btn
                        color="primary"
                        size="lg"
                        text-color="white"
                        @click="btn.clicked"
                    >
                        <div class="text-center">
                            
                            <q-icon size="4em" :name="btn.icon" />
                            <div style="white-space: pre;">{{ btn.label }}</div>
                        </div>
                    </q-btn>
                </div>
            </div>
            <div v-if="boolean_dialog_list.map((e) => e.id).includes(dialog)" class="row q-col-gutter-x-lg" >
                <div class="q-mt-xl">
                    <q-icon class="cursor-pointer" style="vertical-align: center;" name="cancel" size="3em" @click="closeDialog"></q-icon>
                </div>
                <div class="text-h5 q-mt-xl">{{ boolean_dialog_list.find((e) => e.id === dialog).msg }}</div>
                <div class="">
                    <q-btn
                        color="primary"
                        size="xl"
                        text-color="white"
                        @click="boolean_dialog_list.find((e) => e.id === dialog).onOk"
                    >
                        <div class="text-center">
                            <q-icon size="4em" name="circle" />
                            <div>예</div>
                        </div>
                    </q-btn>
                </div>
                <div class="">
                    <q-btn
                        color="primary"
                        size="xl"
                        text-color="white"
                        @click="boolean_dialog_list.find((e) => e.id === dialog).onNo"
                    >
                        <div class="text-center">
                            <q-icon size="4em" name="close" />
                            <div>아니오</div>
                        </div>
                    </q-btn>
                </div>
            </div>
            <div v-if="notice_dialog_list.map((e) => e.id).includes(dialog)" class="row q-col-gutter-x-lg" >
                <div>
                    <q-icon style="vertical-align: center;" name="check" size="3em"></q-icon>
                </div>
                <div class="text-h5">{{ notice_dialog_list.find((e) => e.id === dialog).msg }}</div>
            </div>
            <div v-if="order_dialog_list.map((e) => e.id).includes(dialog)" class="row q-col-gutter-x-lg" >
                <div>
                    <q-icon style="vertical-align: center;" name="cancel" size="3em" @click="closeDialog"></q-icon>
                </div>
                <div class="text-h5">{{ order_dialog_list.find((e) => e.id === dialog).msg }}</div>
            </div>
        </div>
        <q-dialog v-model="showQuitRobotDialog" persistent>
            <div v-if="quitRobotStatus === 'ask'">
                <div class="text-white text-h4 q-mb-lg">로봇을 종료할까요?</div>
                <div class="text-center q-gutter-x-lg">
                    <q-btn color="blue" text-color="white" size="lg" style="width: 100px" @click="quitRobot">예</q-btn>
                    <q-btn color="white" text-color="black" size="lg" @click="showQuitRobotDialog = false" style="width: 100px">아니오</q-btn>
                </div>
            </div>
            <div v-else-if="quitRobotStatus === 'quitting'">
                <div class="text-white text-h4 q-mb-lg">로봇이 복귀한 후 자동으로 종료됩니다.</div>
            </div>
            <div v-else-if="quitRobotStatus === 'quitted'">
                <div class="text-white text-h4 q-mb-lg">로봇이 성공적으로 종료되었습니다.</div>
            </div>
        </q-dialog>

        <!-- CF (Corrective Feedback) 단계별 다이얼로그 -->
        <!-- Step 1: 타입 선택 -->
        <q-dialog :model-value="cfStep === 'select_type'" persistent>
            <q-card style="min-width: 400px">
                <q-card-section class="bg-primary text-white">
                    <div class="text-h6">
                        <q-icon name="pan_tool" class="q-mr-sm" />
                        Corrective Feedback (1/3)
                    </div>
                    <div class="text-caption">어떤 수정이 필요한가요?</div>
                </q-card-section>

                <q-card-section class="q-pt-md">
                    <div class="q-gutter-sm">
                        <q-btn
                            v-for="btn in cf_btns"
                            :key="btn.value"
                            :color="btn.color"
                            text-color="white"
                            class="full-width q-mb-sm"
                            size="lg"
                            @click="selectCFType(btn.value)"
                        >
                            <q-icon :name="btn.icon" class="q-mr-sm" />
                            {{ btn.label }}
                        </q-btn>
                    </div>
                </q-card-section>

                <q-card-actions align="right">
                    <q-btn flat label="취소" color="grey" @click="cancelCF" />
                </q-card-actions>
            </q-card>
        </q-dialog>

        <!-- Step 2: 픽셀 선택은 다이얼로그 없이 상태창에서 안내 (화면 클릭 가능하도록) -->

        <!-- Step 3: 상태 선택 -->
        <q-dialog :model-value="cfStep === 'select_status'" persistent>
            <q-card style="min-width: 400px">
                <q-card-section class="bg-blue text-white">
                    <div class="text-h6">
                        <q-icon name="checklist" class="q-mr-sm" />
                        Corrective Feedback (3/3)
                    </div>
                    <div class="text-caption">
                        선택된 타입: {{ cfSelectedType }}
                        <span v-if="lastClickedPixel.x !== null">
                            | 위치: ({{ lastClickedPixel.x }}, {{ lastClickedPixel.y }})
                        </span>
                    </div>
                </q-card-section>

                <q-card-section class="q-pt-md">
                    <div class="text-subtitle1 q-mb-md">토마토 상태를 선택하세요</div>
                    <div class="q-gutter-sm">
                        <q-btn
                            v-for="opt in getCurrentStatusOptions()"
                            :key="opt.value"
                            :color="opt.color"
                            text-color="white"
                            class="full-width q-mb-sm"
                            size="lg"
                            @click="selectCFStatus(opt.value)"
                        >
                            {{ opt.label }}
                        </q-btn>
                    </div>
                </q-card-section>

                <q-card-actions align="right">
                    <q-btn flat label="취소" color="grey" @click="cancelCF" />
                </q-card-actions>
            </q-card>
        </q-dialog>
    </q-page>
</template>

<script setup>

import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router'
import { startSocket, sendCommand, setMessageHandler } from 'src/utils/websocket';

const $router = useRouter() 
const ctr_mode = ref([])
const is_playing = ref(false)

const ws = startSocket()

const ctr_modes = [
    { 'label': '말하기', icon: 'mic', value: 'voice', clicked: () => ctr_mode.value = 'voice' },
    { 'label': '손짓하기', icon: 'waving_hand', value: 'gesture', clicked: () => ctr_mode.value = 'gesture' },
    { 'label': '조작하기', icon: 'sports_esports', value: 'controller', clicked: () => ctr_mode.value = 'controller' },
]

const robot_btns = [
    { 'label': '수확시작', icon: 'play_arrow', value: 'play', color: 'primary', show: () => !is_playing.value },
    { 'label': '일시정지', icon: 'pause', value: 'pause', color: 'primary', show: () => is_playing.value },
    { 'label': '도와주기', icon: 'build', value: 'correct', color: 'red', show: () => true },
    { 'label': '수확완료', icon: 'done_outline', value: 'finish', color: 'blue', show: () => true },
    { 'label': '도와주기2', icon: 'pan_tool', value: 'cf_trigger', color: 'orange', show: () => true },
]

const rosCameraMain = ref(null)
const rosCameraSub = ref(null)

const currentMain = ref('cam1')

// ============================================================
// Corrective Feedback (CF) 관련
// ============================================================
// CF 단계: 'idle' → 'select_type' → 'select_pixel' → 'select_status' → 전송
const cfStep = ref('idle')
const cfSelectedType = ref(null)
const cfSelectedStatus = ref(null)
const lastClickedPixel = ref({ x: null, y: null, camera: null })
const cfWarningMsg = ref('')  // 유효 영역 외 클릭 시 경고 메시지

// CF 타입별 설정
const cf_btns = [
    { label: '토마토 삭제', icon: 'delete', value: 'DELETE_TOMATO', color: 'red', needsPixel: true, needsStatus: false },
    { label: '토마토 추가', icon: 'add_circle', value: 'ADD_TOMATO', color: 'green', needsPixel: true, needsStatus: true },
    { label: '상태 수정', icon: 'edit', value: 'UPDATE_CLASS', color: 'orange', needsPixel: true, needsStatus: true },
    { label: '스캔결과 수정', icon: 'refresh', value: 'UPDATE_SCAN_RESULT', color: 'blue', needsPixel: false, needsStatus: true },
]

// Pre-pick 상태 옵션 (DELETE/ADD/UPDATE_CLASS)
const prePickStatusOptions = [
    { label: '익음 (Ripe)', value: 0, color: 'green' },
    { label: '안익음 (Unripe)', value: 1, color: 'yellow' },
]

// Post-pick 상태 옵션 (UPDATE_SCAN_RESULT)
const postPickStatusOptions = [
    { label: '익음 (Ripe)', value: 0, color: 'green' },
    { label: '썩음 (Rotten)', value: 2, color: 'red' },
]

const status_list = [
    { label: '대기중', value: 'pending', msg: '수확을 시작할까요?', color: 'green-2', availabe_btns: ['play'] },
    { label: '작업진행', value: 'moving', msg: '작업을 위해 이동중이에요!', color: 'yellow-2', availabe_btns: ['play', 'pause'] },
    { label: '작업진행', value: 'picking', msg: '로봇이 토마토를 수확하고 있어요!', color: 'yellow-2', availabe_btns: ['play', 'correct', 'pause', 'finish'] },
    { label: '작업진행', value: 'checking', msg: '로봇이 토마토를 확인하고 있어요!', color: 'yellow-2', availabe_btns: ['play', 'correct', 'pause', 'finish'] },
    { label: '작업진행', value: 'placing', msg: '로봇이 토마토를 바구니에 넣고 있어요!', color: 'yellow-2', availabe_btns: ['play', 'pause', 'finish'] },
    { label: '작업진행', value: 'discarding', msg: '로봇이 토마토를 폐기하고 있어요!', color: 'yellow-2', availabe_btns: ['play', 'pause', 'finish'] },
    { label: '수확수정중', value: 'correcting_pick', msg: '도와주세요! 이 토마토를 어떻게 하면 좋을까요?', color: 'yellow-2', availabe_btns: [] },
    { label: '수확수정중', value: 'correcting_check', msg: '도와주세요! 이 토마토를 어떻게 하면 좋을까요?', color: 'yellow-2', availabe_btns: [] },
    { label: '작업완료', value: 'finished', msg: '모든 작업을 완료했습니다! 추가 작업이 있다면 알려주세요!', color: 'purple-2', availabe_btns: ['play'] },
]
const status = ref(status_list[0])

const dialog = ref(null)

function switchCam() {
    let tmpSrcObject = null
    tmpSrcObject = rosCameraMain.value.srcObject
    rosCameraMain.value.srcObject = rosCameraSub.value.srcObject
    rosCameraSub.value.srcObject = tmpSrcObject

    if (currentMain.value === 'cam1') {
        currentMain.value = 'cam2'
    } else {
        currentMain.value = 'cam1'
    }
}

// const boolean_dialog_list = [
//     { id: 'finish', msg: '작업을 완료할까요?', onOk: () => { statusBuffer.value = status.value; status.value = status_list.find((e) => e.value === 'finished'); dialog.value = null; }, onNo: closeDialog },
//     { id: 'stop', msg: '작업을 일시중지할까요?', onOk: () => { statusBuffer.value = status.value; status.value = status_list.find((e) => e.value === 'paused'); dialog.value = null; }, onNo: closeDialog },
//     { id: 'harvest', msg: '이 토마토를 수확할까요?', onOk: harvest_tomato, onNo: dont_harvest_tomato },
//     { id: 'discard', msg: '이 토마토를 버릴까요?', onOk: discard_tomato, onNo: dont_discard_tomato },
//     { id: 'missed_tomato', msg: '놓친 토마토가 있나요?', onOk: finding_missed_tomato, onNo: dont_find_missed_tomato },
// ]

const notice_dialog_list = [
    { id: 'harvest_true', msg: '지금부터 토마토 수확을 시작할게요!' },
    { id: 'harvest_false', msg: '네, 이 토마토는 수확하지 않을게요!' },
    { id: 'discard_true', msg: '이 토마토를 버렸어요!' },
    { id: 'discard_false', msg: '이 토마토는 잘 수확했어요!' },
    { id: 'missed_tomato_true', msg: '확인했어요! 이 토마토를 수확하도록 하겠습니다!' },
    { id: 'missed_tomato_false', msg: '네, 잘 수확하도록 하겠습니다!' },
]

const order_dialog_list = [
    { id: 'find_missed_tomato', msg: '화면에서 놓친 토마토가 있는 부분을 눌러주세요!' },
]

const statusBuffer = ref(null)
function clickRobotBtn(value) {
    if (value === 'correct') {
        sendCommand(ws, 'start_robot')
    } else if (value === 'pause') {
        sendCommand(ws, 'pause')
    } else if (value === 'cf_trigger') {
        // CF 시작: cf_start 전송 + 타입 선택 단계로
        startCF()
    } else if (status.value.value === 'pending') {
        sendCommand(ws, 'start_harvest')
    }
}

// ============================================================
// Corrective Feedback 단계별 플로우
// ============================================================
// 플로우: 도와주기2 → cf_start → 타입선택 → (픽셀선택) → (상태선택) → cf_action

function startCF() {
    // CF 시작: BT에 cf_start 전송
    sendCommand(ws, 'cf_start', { robot_name: 'brl_robot' })
    console.log('[CF] Started - cf_start sent')

    // 상태 초기화 & 타입 선택 단계로
    cfStep.value = 'select_type'
    cfSelectedType.value = null
    cfSelectedStatus.value = null
    lastClickedPixel.value = { x: null, y: null, camera: null }
}

function selectCFType(cfType) {
    cfSelectedType.value = cfType
    const cfConfig = cf_btns.find(btn => btn.value === cfType)
    console.log(`[CF] Type selected: ${cfType}`)

    if (cfConfig.needsPixel) {
        // 픽셀 선택 필요 → 픽셀 선택 단계로
        cfStep.value = 'select_pixel'
    } else if (cfConfig.needsStatus) {
        // 픽셀 불필요, 상태 선택 필요 → 상태 선택 단계로
        cfStep.value = 'select_status'
    } else {
        // 둘 다 불필요 → 바로 전송
        sendCFAction()
    }
}

function onCFPixelSelected() {
    // 픽셀 선택 완료 후 호출됨 (videoClicked에서)
    const cfConfig = cf_btns.find(btn => btn.value === cfSelectedType.value)
    console.log(`[CF] Pixel selected: (${lastClickedPixel.value.x}, ${lastClickedPixel.value.y})`)

    if (cfConfig.needsStatus) {
        // 상태 선택 필요 → 상태 선택 단계로
        cfStep.value = 'select_status'
    } else {
        // 상태 선택 불필요 → 바로 전송
        sendCFAction()
    }
}

function selectCFStatus(statusValue) {
    cfSelectedStatus.value = statusValue
    console.log(`[CF] Status selected: ${statusValue}`)

    // 모든 선택 완료 → 전송
    sendCFAction()
}

function sendCFAction() {
    const cfParam = {
        action: cfSelectedType.value,
        pixel_x: lastClickedPixel.value.x || 0,
        pixel_y: lastClickedPixel.value.y || 0,
        camera: lastClickedPixel.value.camera || 'camera1',
        robot_name: 'brl_robot',
        new_class: cfSelectedStatus.value !== null ? String(cfSelectedStatus.value) : ''
    }

    console.log('[CF] Sending cf_action:', cfParam)
    sendCommand(ws, 'cf_action', cfParam)

    // CF 종료
    resetCF()
}

function cancelCF() {
    // CF 취소: cf_cancel 전송
    sendCommand(ws, 'cf_cancel', { robot_name: 'brl_robot' })
    console.log('[CF] Cancelled')
    resetCF()
}

function resetCF() {
    cfStep.value = 'idle'
    cfSelectedType.value = null
    cfSelectedStatus.value = null
    lastClickedPixel.value = { x: null, y: null, camera: null }
}

// 현재 상태 옵션 가져오기
function getCurrentStatusOptions() {
    if (cfSelectedType.value === 'UPDATE_SCAN_RESULT') {
        return postPickStatusOptions
    }
    return prePickStatusOptions
}

function closeDialog() {
    dialog.value = null;
    status.value = statusBuffer.value;
}

const showQuitRobotDialog = ref(false)
const quitRobotStatus = ref('ask')

function quitRobot() {
    quitRobotStatus.value = 'quitting'
    sendCommand(ws, 'stop_robot')
    setTimeout(() => {
        quitRobotStatus.value = 'quitted'
        setTimeout(() => {
            $router.push('/')
        }, 5000)
    }, 2500)
}

function videoClicked(event) {
    // ============================================================
    // CF용 픽셀 좌표 저장 (cam1 main일 때만)
    // ============================================================
    // 화면에 표시된 video 크기와 원본 비디오 해상도가 다를 수 있음
    // 예: 원본 640x480이 CSS로 800x600으로 표시됨
    //
    // 스케일링 공식:
    //   실제_x = 클릭_x * (원본_width / 표시_width)
    //   실제_y = 클릭_y * (원본_height / 표시_height)
    //
    // video.videoWidth/Height: 원본 비디오 해상도 (고정)
    // rect.width/height: 현재 화면 표시 크기 (동적)
    // → 창 크기 변경, CSS 변경 시에도 자동으로 올바른 좌표 계산
    // ============================================================

    // CF 픽셀 선택 모드가 아니면 무시 (기존 로직만 실행)
    if (cfStep.value !== 'select_pixel') {
        // 기존 로직 유지
        if (status.value.value === 'finding_missed_tomato') {
            dialog.value = 'missed_tomato_true'
            status.value = status_list.find((e) => e.value === 'harvesting')
            setTimeout(() => {
                dialog.value = null
                status.value = status_list.find((e) => e.value === 'harvesting')
            }, 5000)
        }
        return
    }

    // CF 픽셀 선택 모드: cam1 main일 때만 좌표 저장
    if (currentMain.value !== 'cam1') {
        console.log('[CF] cam1이 main일 때만 좌표 선택 가능합니다')
        return
    }

    const video = event.target
    const rect = video.getBoundingClientRect()

    // 화면 좌표 (CSS 크기 기준)
    const displayX = event.clientX - rect.left
    const displayY = event.clientY - rect.top

    // 원본 비디오 해상도로 스케일링
    const scaleX = video.videoWidth / rect.width
    const scaleY = video.videoHeight / rect.height

    const pixelX = Math.round(displayX * scaleX)
    const pixelY = Math.round(displayY * scaleY)

    // =========================================================================
    // CF Workspace 유효 영역 설정
    // =========================================================================
    // 카메라 원본 해상도 640x480 기준 유효 클릭 영역
    // 로봇 작업 공간에 맞게 조정 필요 (현재: 하베스터 기준)
    // - minX: 왼쪽 경계 (로봇 팔 도달 범위)
    // - maxX: 오른쪽 경계 (이미지 끝)
    // - minY: 상단 경계 (너무 높은 토마토 제외)
    // - maxY: 하단 경계 (바닥/장애물 제외)
    // =========================================================================
    const WORKSPACE = { minX: 280, maxX: 640, minY: 60, maxY: 450 }
    if (pixelX < WORKSPACE.minX || pixelX > WORKSPACE.maxX ||
        pixelY < WORKSPACE.minY || pixelY > WORKSPACE.maxY) {
        cfWarningMsg.value = '영역 안을 선택해주세요'
        setTimeout(() => { cfWarningMsg.value = '' }, 1000)
        console.log(`[CF] 유효 영역 외 클릭: (${pixelX}, ${pixelY})`)
        return
    }

    lastClickedPixel.value = {
        x: pixelX,
        y: pixelY,
        camera: 'camera1'
    }
    console.log(`[CF] Pixel selected: (${pixelX}, ${pixelY})`)

    // 다음 단계로 진행
    onCFPixelSelected()
}

// function changeStatus(status) {
//     console.log(status)
//     ws.send(JSON.stringify({command: "change_status", param: status}))
// }

onMounted(() => {
    // Use setMessageHandler for reconnect-safe message handling
    setMessageHandler((event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === 'data' && !['correcting_pick', 'correcting_check'].includes(status.value.value)) {
            const data = JSON.parse(msg.value)
            status.value = status_list.find((e) => e.value === data.status)
            is_playing.value = Boolean(data.playing)
        }
    })
})


</script>
  
