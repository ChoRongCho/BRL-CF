<template>
    <q-page class="row q-mx-lg">
        <!-- 이동 중 또는 확인 중일 때 화면 어둡게 처리 및 안내 문구 표시 -->
        <div v-if="status.value === 'preparing_nav'"
             class="fullscreen flex flex-center"
             style="background: rgba(0, 0, 0, 0.5); z-index: 5000; pointer-events: none; transition: background 0.3s;">
             <div class="text-white text-center">
                    <div class="text-h3 text-bold q-mb-md" v-if="!isReturning">로봇이 이동을 준비중이에요.</div>
                    <div class="text-h3 text-bold q-mb-md" v-else>제자리로 복귀 중이에요.</div>
                    <div class="text-h4">잠시만 기다려주세요!</div>
                    <q-spinner-dots size="4em" class="q-mt-xl" />
             </div>
        </div>

        <div class="col-2">
            <div class="q-pa-md row">
                <div class="col text-center" @click="showQuitRobotDialog = true">
                    <!-- <q-icon name="power_settings_new" size="xl"></q-icon> -->
                    <img src="~assets/power_icon.svg" style="width: 56px; height: 56px;" />
                    <div>시스템 종료</div>
                </div>
                <!-- <div class="col text-center">
                    <q-icon name="settings" size="xl"></q-icon>
                    <div>로봇관리</div>
                </div> -->
            </div>
            <div class="q-pa-md text-center">
                <div class="text-h6 text-bold">
                    <!-- 작업 정보 -->
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
                    <!-- 조작 방법 -->
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
            <!-- 테스트용 상태 변경 드롭다운 -->
            <div class="q-pa-md">
                <div class="text-caption text-grey q-mb-xs">
                    <q-icon name="bug_report" size="xs" class="q-mr-xs" />
                    DEV: 상태 변경
                </div>
                <q-select
                    :model-value="status"
                    :options="status_list"
                    option-label="label"
                    option-value="value"
                    dense
                    outlined
                    :display-value="`${status.label} (${status.value})`"
                    @update:model-value="onDevStatusChange"
                >
                    <template v-slot:option="scope">
                        <q-item v-bind="scope.itemProps">
                            <q-item-section>
                                <q-item-label>{{ scope.opt.label }}</q-item-label>
                                <q-item-label caption>{{ scope.opt.value }}</q-item-label>
                            </q-item-section>
                        </q-item>
                    </template>
                </q-select>
            </div>
        </div>
        <div class="col q-pa-md q-px-xl">
            <!-- 상태창 -->
            <div>
                <div class="bg-white text-h6 text-bold q-pa-sm" style="border: 3px solid #02542D;" >
                <q-chip :color="status.color" text-color="black">{{ status.label }}</q-chip>
                {{ status.msg }}</div>
            </div>

            <div class="q-pt-lg">
                <div style="position: relative;">
                    <video
                        ref="rosCameraMain"
                        autoplay playsinline muted
                        style=" width: 100%; aspect-ratio: 4 / 3; background-color: black;"
                        :style="status.value === 'finding_missed_tomato' ? 'cursor: pointer; border: 6px solid #0000FF' : 'border: 3px solid #02542D;'"
                        @click="videoClicked"
                    ></video>

                    <!-- ask_class: 대상 토마토 바운딩 박스 하이라이트 -->
                    <div v-if="askClassBoxStyle"
                         :style="askClassBoxStyle"
                         class="ask-class-bbox" />

                    <!-- navigating 상태: 카메라 영역 위 오버레이 -->
                    <div v-if="status.value === 'navigating'"
                         class="absolute-full flex flex-center"
                         style="background: rgba(100, 100, 100, 0.7); z-index: 10; pointer-events: none;">
                        <div class="text-white text-center">
                            <div class="text-h4 text-bold q-mb-md" v-if="!isReturning">로봇이 이동중이에요.</div>
                            <div class="text-h4 text-bold q-mb-md" v-else>제자리로 복귀 중이에요.</div>
                            <div class="text-h5">잠시만 기다려주세요!</div>
                            <q-spinner-dots size="3em" class="q-mt-lg" />
                        </div>
                    </div>

                    <!-- scanning / detecting 상태: 카메라 영역 위 오버레이 -->
                    <div v-if="status.value === 'scanning' || status.value === 'detecting'"
                         class="absolute-full flex flex-center"
                         style="background: rgba(100, 100, 100, 0.7); z-index: 10; pointer-events: none;">
                        <div class="text-white text-center">
                            <div class="text-h5 text-bold q-mb-md">잠시만 기다려주세요!</div>
                            <div class="text-h4">로봇이 토마토의 상태를 확인하고 있어요!</div>
                            <q-spinner-dots size="3em" class="q-mt-lg" />
                        </div>
                    </div>

                    <!-- task_planning 상태: 카메라 영역 위 오버레이 -->
                    <div v-if="status.value === 'task_planning'"
                         class="absolute-full flex flex-center"
                         style="background: rgba(100, 100, 100, 0.7); z-index: 10; pointer-events: none;">
                        <div class="text-white text-center">
                            <div class="text-h5 text-bold q-mb-md">잠시만 기다려주세요!</div>
                            <div class="text-h4">로봇이 작업 계획 중입니다!</div>
                            <q-spinner-dots size="3em" class="q-mt-lg" />
                        </div>
                    </div>

                    <!-- Checking 단계 통합 인터랙션 UI (영상 하단 오버레이) -->
                    <transition name="fade">
                        <div v-if="interactionState !== 'none' && interactionState !== 'guide' && interactionState !== 'ask_confirm_delete' && interactionState !== 'ask_confirm_add' && interactionState !== 'select_held_class' && interactionState !== 'ask_class' && interactionState !== 'ask_picking_class'"
                             class="absolute-bottom text-center q-ma-md"
                             style="z-index: 100;">
                            <div class="inline-block bg-primary text-white q-pa-md rounded-borders shadow-5"
                                 style="opacity: 0.95; border: 2px solid white; min-width: 400px;">

                                <!-- 2. 삭제/추가 선택 -->
                                <div v-if="interactionState === 'select_action'">
                                    <div class="text-h5 text-bold q-mb-sm">
                                        <q-icon name="touch_app" size="sm" class="q-mr-sm"/>
                                        이 위치에서 무엇을 하시겠습니까?
                                    </div>
                                    <div class="q-gutter-x-md">
                                        <q-btn color="red" text-color="white" label="토마토 삭제"
                                               @click="handleInteractionResponse('delete')" />
                                        <q-btn color="green" text-color="white" label="토마토 추가"
                                               @click="handleInteractionResponse('add')" />
                                        <q-btn outline color="white" label="취소"
                                               @click="handleInteractionResponse('cancel')" />
                                    </div>
                                </div>

                                <!-- 3. 클래스 선택 -->
                                <div v-else-if="interactionState === 'select_class'">
                                    <div class="text-h5 text-bold q-mb-sm">
                                        <q-icon name="category" size="sm" class="q-mr-sm"/>
                                        토마토 상태를 선택하세요
                                    </div>
                                    <div class="q-gutter-x-md">
                                        <q-btn color="green" text-color="white" label="익음"
                                               @click="handleInteractionResponse('ripe')" />
                                        <q-btn color="yellow" text-color="black" label="안익음"
                                               @click="handleInteractionResponse('unripe')" />
                                        <q-btn outline color="white" label="취소"
                                               @click="handleInteractionResponse('cancel')" />
                                    </div>
                                </div>

                                <!-- 4. 자동 완료 예고 메시지 -->
                                <div v-else-if="interactionState === 'auto_finish'">
                                    <div class="text-h5 text-bold">
                                        <q-icon name="check_circle" size="sm" class="q-mr-sm"/>
                                        화면 확인이 끝났습니다!<br>지금부터 수확을 시작할게요!
                                    </div>
                                </div>

                                <!-- 5. 썩음 여부 확인 질문 -->
                                <div v-else-if="interactionState === 'ask_rotten'">
                                    <div class="text-h5 text-bold q-mb-sm">
                                        <q-icon name="help_outline" size="sm" class="q-mr-sm"/>
                                        이 토마토가 썩었나요?
                                    </div>
                                    <div class="q-gutter-x-md">
                                        <q-btn color="white" text-color="red" label="예" @click="handleInteractionResponse('rotten_yes')" style="width: 100px; font-weight: bold;" />
                                        <q-btn outline color="white" label="아니오" @click="handleInteractionResponse('rotten_no')" style="width: 100px;" />
                                    </div>
                                </div>

                            </div>
                        </div>
                    </transition>

                    <!-- ask_detecting 인터랙션: 비디오 하단 30% 오버레이 -->
                    <transition name="fade">
                        <div v-if="interactionState === 'guide' || interactionState === 'ask_confirm_delete' || interactionState === 'ask_confirm_add' || interactionState === 'select_held_class' || interactionState === 'ask_class' || interactionState === 'ask_picking_class'"
                             style="position: absolute; bottom: 0; left: 0; right: 0; height: 30%;
                                    background: rgba(0, 0, 0, 0.65); z-index: 15;
                                    display: flex; align-items: center; justify-content: center;"
                             :style="interactionState === 'guide' ? 'pointer-events: none;' : ''">
                            <div class="text-white text-center">
                                <!-- 기본 가이드 메시지 -->
                                <template v-if="interactionState === 'guide'">
                                    <div class="text-h5 text-bold q-mb-lg">
                                        <q-icon name="touch_app" size="sm" class="q-mr-sm"/>
                                        <span v-if="status.value === 'correct_picking'">
                                            변경하고 싶은 토마토를 클릭해주세요
                                        </span>
                                        <span v-else>
                                            놓친 토마토나 잘못 확인한 토마토를 화면에서 눌러주세요
                                        </span>
                                    </div>
                                    <q-btn color="blue" text-color="white" size="lg"
                                           :label="pendingCFActions.length > 0
                                               ? `확인 완료 (${pendingCFActions.length}건 수정)`
                                               : '확인 완료'"
                                           @click="finishCorrecting" style="width: 220px; pointer-events: auto;" />
                                </template>
                                <!-- 토마토 삭제 확인 -->
                                <template v-else-if="interactionState === 'ask_confirm_delete'">
                                    <div class="text-h4 text-bold q-mb-lg">
                                        <q-icon name="help_outline" size="md" class="q-mr-sm"/>
                                        여기에 토마토가 없나요?
                                    </div>
                                    <div class="q-gutter-x-lg">
                                        <q-btn color="red" text-color="white" label="예" size="lg"
                                               @click="handleInteractionResponse('confirm_delete_yes')" style="width: 120px; font-weight: bold;" />
                                        <q-btn color="white" text-color="black" label="아니오" size="lg"
                                               @click="handleInteractionResponse('cancel')" style="width: 120px;" />
                                    </div>
                                </template>
                                <!-- 토마토 추가 확인 -->
                                <template v-else-if="interactionState === 'ask_confirm_add'">
                                    <div class="text-h4 text-bold q-mb-lg">
                                        <q-icon name="help_outline" size="md" class="q-mr-sm"/>
                                        여기에 토마토가 있나요?
                                    </div>
                                    <div class="q-gutter-x-lg">
                                        <q-btn color="green" text-color="white" label="예" size="lg"
                                               @click="handleInteractionResponse('confirm_add_yes')" style="width: 120px; font-weight: bold;" />
                                        <q-btn color="white" text-color="black" label="아니오" size="lg"
                                               @click="handleInteractionResponse('cancel')" style="width: 120px;" />
                                    </div>
                                </template>
                                <!-- 들고 있는 토마토 썩음 여부 (correct_placing) -->
                                <template v-else-if="interactionState === 'select_held_class'">
                                    <div class="text-h4 text-bold q-mb-lg">
                                        <q-icon name="help_outline" size="md" class="q-mr-sm"/>
                                        이 토마토가 썩었나요?
                                    </div>
                                    <div class="q-gutter-x-lg">
                                        <q-btn color="red" text-color="white" label="예" size="lg"
                                               @click="handleHeldTomatoClass(2)" style="width: 120px; font-weight: bold;" />
                                        <q-btn color="white" text-color="black" label="아니오" size="lg"
                                               @click="cancelCorrectPlacing" style="width: 120px;" />
                                    </div>
                                </template>
                                <!-- correct_picking: 토마토 클릭 후 익음/안익음 선택 -->
                                <template v-else-if="interactionState === 'ask_picking_class'">
                                    <div class="text-h4 text-bold q-mb-lg">
                                        <q-icon name="help_outline" size="md" class="q-mr-sm"/>
                                        이 토마토를 수확할까요?
                                    </div>
                                    <div class="q-gutter-x-lg">
                                        <q-btn color="green" text-color="white" label="예" size="lg"
                                               @click="handlePickingClassResponse(0)" style="width: 120px; font-weight: bold;" />
                                        <q-btn color="white" text-color="black" label="아니오" size="lg"
                                               @click="handlePickingClassResponse(1)" style="width: 120px;" />
                                    </div>
                                </template>
                                <!-- ask_class: KB에서 confidence 낮은 토마토 확인 요청 (스캔 결과: 썩음/안썩음) -->
                                <template v-else-if="interactionState === 'ask_class' && askClassData && askClassData.pixel_x === 0 && askClassData.pixel_y === 0">
                                    <div class="text-h4 text-bold q-mb-lg">
                                        <q-icon name="help_outline" size="md" class="q-mr-sm"/>
                                        이 토마토가 썩었나요?
                                    </div>
                                    <div class="q-gutter-x-lg">
                                        <q-btn color="red" text-color="white" label="예" size="lg"
                                               @click="handleAskClassResponse(2)" style="width: 120px; font-weight: bold;" />
                                        <q-btn color="white" text-color="black" label="아니오" size="lg"
                                               @click="handleAskClassResponse(0)" style="width: 120px;" />
                                    </div>
                                </template>
                                <!-- ask_class: KB에서 confidence 낮은 토마토 확인 요청 (감지 결과: 익음/안익음) -->
                                <template v-else-if="interactionState === 'ask_class' && askClassData">
                                    <div class="text-h4 text-bold q-mb-lg">
                                        <q-icon name="help_outline" size="md" class="q-mr-sm"/>
                                        이 토마토를 수확할까요?
                                    </div>
                                    <div class="q-gutter-x-lg">
                                        <q-btn color="green" text-color="white" label="예" size="lg"
                                               @click="handleAskClassResponse(0)" style="width: 120px; font-weight: bold;" />
                                        <q-btn color="white" text-color="black" label="아니오" size="lg"
                                               @click="handleAskClassResponse(1)" style="width: 120px;" />
                                    </div>
                                </template>
                            </div>
                        </div>
                    </transition>

                    <div class="absolute-top-right" style="width: 33%;">
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
                    :color="isBtnActive(btn.value) ? 'green' : 'secondary'"
                    size="lg"
                    :text-color="isBtnActive(btn.value) ? 'white' : 'black'"
                    :disable="!isBtnAvailable(btn.value)"
                    @click="clickRobotBtn(btn.value)"
                >
                    <div class="column text-center">
                        <q-icon center size="4em" :name="btn.icon" :color="isBtnActive(btn.value) ? 'white' : (isBtnAvailable(btn.value) ? btn.color : 'grey-6')" />
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
        <q-dialog v-model="factQueryVisible" persistent>
            <q-card style="min-width: 520px; max-width: 720px;">
                <q-card-section class="text-center">
                    <div class="text-h5 text-bold q-mb-md">확인이 필요합니다</div>
                    <div class="text-h4 q-mb-sm">{{ pendingFactQuery?.question }}</div>
                    <div class="text-caption text-grey-7">{{ pendingFactQuery?.fact }}</div>
                </q-card-section>
                <q-card-actions align="center" class="q-pb-lg q-gutter-x-lg">
                    <q-btn color="positive" size="lg" label="예"
                           @click="answerFactQuery('true')" />
                    <q-btn color="negative" size="lg" label="아니요"
                           @click="answerFactQuery('false')" />
                </q-card-actions>
            </q-card>
        </q-dialog>

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

    </q-page>
</template>

<script setup>

import { ref, computed, onMounted } from 'vue';
import { useQuasar } from 'quasar'
import { useRouter } from 'vue-router'
import { startSocket, sendCommand, setMessageHandler } from 'src/utils/websocket';
import doneIcon from 'assets/done_icon.svg'
import handIcon from 'assets/hand_icon.svg'
import helpIcon from 'assets/help_icon.svg'
import robotIcon from 'assets/robot_icon.svg'
import speakingIcon from 'assets/speaking_icon.svg'
import startIcon from 'assets/start_icon.svg'



const $q = useQuasar()
const $router = useRouter() 
const ctr_mode = ref([])
const is_playing = ref(false)

const ws = startSocket()
const factQueryVisible = ref(false)
const pendingFactQuery = ref(null)
const factQueryTimeout = ref(null)

function clearFactQuery() {
    if (factQueryTimeout.value) {
        clearTimeout(factQueryTimeout.value)
        factQueryTimeout.value = null
    }
    factQueryVisible.value = false
    pendingFactQuery.value = null
}

function answerFactQuery(answer) {
    if (!pendingFactQuery.value) return
    sendCommand(ws, 'query_fact_response', {
        request_id: pendingFactQuery.value.request_id,
        answer: answer
    })
    clearFactQuery()
}

const ctr_modes = [
    { 'label': '말하기', icon: 'img:' + speakingIcon, value: 'voice', clicked: () => ctr_mode.value = 'voice' },
    { 'label': '손짓하기', icon: 'img:' + handIcon, value: 'gesture', clicked: () => ctr_mode.value = 'gesture' },
    { 'label': '조작하기', icon: 'img:' + robotIcon, value: 'controller', clicked: () => ctr_mode.value = 'controller' },
]

const robot_btns = [
    { 'label': '수확시작', icon: 'img:' + startIcon, value: 'play', color: 'primary' },
    { 'label': '도와주기', icon: 'img:' + helpIcon, value: 'correct', color: 'red' },
    { 'label': '수확완료', icon: 'img:' + doneIcon, value: 'finish', color: 'blue' },
]

function isBtnActive(value) {
    if (value === 'play') return is_playing.value && !['ask_detecting', 'correct_picking', 'correct_placing', 'correct_proactive'].includes(status.value.value)
    if (value === 'correct') return ['ask_detecting', 'correct_picking', 'correct_placing', 'correct_proactive'].includes(status.value.value)
    if (value === 'finish') return isReturning.value
    return false
}

function isBtnAvailable(value) {
    if (value === 'play') return status.value.availabe_btns.includes('play') || status.value.availabe_btns.includes('pause')
    return status.value.availabe_btns.includes(value)
}

const rosCameraMain = ref(null)
const rosCameraSub = ref(null)

const currentMain = ref('cam1')

// ============================================================
// Corrective Feedback (CF) 관련
// ============================================================
const lastClickedPixel = ref({ x: null, y: null, camera: null })
const isReturning = ref(false) // 로봇 복귀 중 여부

// Checking 단계 인터랙션 상태 관리
// state: 'none' | 'guide' | 'select_action' | 'select_class' | 'auto_finish' | 'ask_rotten' | 'select_held_class'
//        | 'ask_confirm_delete' | 'ask_confirm_add'
const interactionState = ref('none')

// Scene Graph 데이터 저장 (토마토 bounding box 정보)
const sceneGraphData = ref(null)

// ask_detecting에서 클릭한 토마토 임시 저장 (삭제 확인용)
const clickedTomato = ref(null)

// ask_detecting: CF 액션 큐잉 (배치 제출용)
// { action: 'ADD_TOMATO'|'DELETE_TOMATO', pixel_x, pixel_y, camera, new_class? }
const pendingCFActions = ref([])

// /cf/ask_class 관련
const askClassData = ref(null)
const askClassTimeout = ref(null)

const askClassBoxStyle = computed(() => {
    if (interactionState.value !== 'ask_class' || !askClassData.value) return null
    const { pixel_x, pixel_y } = askClassData.value
    if (pixel_x === 0 && pixel_y === 0) return null

    const video = rosCameraMain.value
    if (!video) return null

    const sourceW = video.videoWidth || 640
    const sourceH = video.videoHeight || 480

    const boxSize = 12  // 비디오 너비 대비 % (토마토 크기 근사치)
    const boxSizeH = boxSize * (sourceW / sourceH) * (3 / 4)  // 종횡비 보정

    return {
        left: `${(pixel_x / sourceW) * 100 - boxSize / 2}%`,
        top: `${(pixel_y / sourceH) * 100 - boxSizeH / 2}%`,
        width: `${boxSize}%`,
        height: `${boxSizeH}%`,
    }
})

const status_list = [
    { label: '대기중', value: 'active', msg: '무엇을 도와드릴까요?', color: 'green-2', availabe_btns: ['play'] },
    { label: '계획중', value: 'task_planning', msg: '로봇이 작업 계획을 세우고 있어요!', color: 'yellow-2', availabe_btns: [] },
    { label: '이동준비중', value: 'preparing_nav', msg: '로봇이 이동을 준비중이에요!', color: 'yellow-2', availabe_btns: ['play', 'pause'] },
    { label: '로봇이동중', value: 'navigating', msg: '로봇이 이동중이에요!', color: 'yellow-2', availabe_btns: ['play', 'pause'] },
    { label: '스캔중', value: 'scanning', msg: '로봇이 토마토의 상태를 확인하고 있어요!', color: 'yellow-2', availabe_btns: ['pause'] },
    { label: '감지중', value: 'detecting', msg: '로봇이 토마토의 상태를 확인하고 있어요!', color: 'yellow-2', availabe_btns: ['pause'] },
    { label: '감지확인중', value: 'ask_detecting', msg: '토마토 상태를 확인해주세요!', color: 'blue-2', availabe_btns: [] },
    { label: '수확진행중', value: 'picking', msg: '로봇이 토마토를 수확하고 있어요!', color: 'yellow-2', availabe_btns: ['correct', 'pause'] },
    { label: '수확수정중', value: 'correct_picking', msg: '변경하고 싶은 토마토를 클릭해주세요!', color: 'blue-2', availabe_btns: [] },
    { label: '수확진행중', value: 'placing', msg: '로봇이 토마토를 바구니에 넣고 있어요!', color: 'yellow-2', availabe_btns: ['correct', 'pause'] },
    { label: '수확수정중', value: 'correct_placing', msg: '이 토마토가 썩었나요?', color: 'blue-2', availabe_btns: [] },
    { label: '확인요청중', value: 'correct_proactive', msg: '토마토 상태를 확인해주세요!', color: 'blue-2', availabe_btns: [] },
    { label: '일시정지', value: 'halted', msg: '작업을 계속 하려면 [수확 시작]을 눌러주세요!', color: 'red-2', availabe_btns: ['play'] },
    { label: '복귀중', value: 'returning', msg: '로봇이 제자리로 복귀하고 있어요!', color: 'blue-2', availabe_btns: [] },
    { label: '종료완료', value: 'bt_job_done', msg: '로봇이 성공적으로 복귀했습니다.', color: 'green-2', availabe_btns: [] },
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
    { id: ""}
]

const order_dialog_list = [
    { id: 'find_missed_tomato', msg: '화면에서 놓친 토마토가 있는 부분을 눌러주세요!' },
]

const statusBuffer = ref(null)
function clickRobotBtn(value) {
    if (value === 'correct') {
        if (status.value.value === 'picking') {
            // picking → correct_picking 전환
            sendCommand(ws, 'cf_trigger', { robot_name: 'brl_robot' })
            console.log('[도와주기] cf_trigger sent')

            status.value = status_list.find((e) => e.value === 'correct_picking')
            interactionState.value = 'guide'
            console.log('[도와주기] picking → correct_picking')
        } else if (status.value.value === 'placing') {
            // placing → correct_placing 전환
            sendCommand(ws, 'cf_trigger', { robot_name: 'brl_robot' })
            console.log('[도와주기] cf_trigger sent')

            status.value = status_list.find((e) => e.value === 'correct_placing')
            interactionState.value = 'select_held_class'
            console.log('[도와주기] placing → correct_placing')
        }
    } else if (value === 'play') {
        if (is_playing.value) {
            // 일시정지: BT pause 명령 전송
            sendCommand(ws, 'pause')
            status.value = status_list.find((e) => e.value === 'halted')
            is_playing.value = false
        } else if (status.value.value === 'active' || status.value.value === 'halted') {
            // BT resume 명령 전송
            sendCommand(ws, 'start_harvest')
        }
    } else if (value === 'finish') {
        // 수확 완료 -> 복귀 시뮬레이션
        if (status.value.value === 'halted') {
            console.log('[Simulation] Finishing task, returning home...')
            isReturning.value = true
            status.value = status_list.find((e) => e.value === 'navigating')

            // 5초 후 대기 상태(active)로 복귀
            setTimeout(() => {
                status.value = status_list.find((e) => e.value === 'active')
                isReturning.value = false
                is_playing.value = false
                console.log('[Simulation] Returned home. Status: active')
            }, 5000)
        }
    }
}

// ============================================================
// Correcting Check 관련 함수
// ============================================================

// 토마토 존재 여부 확인 함수
function findTomatoAtPixel(pixelX, pixelY) {
    if (!sceneGraphData.value || !Array.isArray(sceneGraphData.value)) {
        return null
    }

    for (const tomato of sceneGraphData.value) {
        const bbox = tomato.properties['bounding_box']
        if (!bbox) continue

        const coords = typeof bbox === 'string'
            ? bbox.replace(/[[\]]/g, '').split(',').map(Number)
            : bbox
        const [x1, y1, x2, y2] = coords

        if (pixelX >= x1 && pixelX <= x2 && pixelY >= y1 && pixelY <= y2) {
            const tomatoClass = tomato.properties['tomato_class']
            const currentClass = tomatoClass !== undefined ? parseInt(tomatoClass) : 0
            return { ...tomato, currentClass }
        }
    }
    return null
}

function sendCorrectingAction(action, newClass = null) {
    const param = {
        action: action,
        pixel_x: lastClickedPixel.value.x || 0,
        pixel_y: lastClickedPixel.value.y || 0,
        camera: lastClickedPixel.value.camera || 'camera1',
        robot_name: 'brl_robot',
        new_class: newClass !== null ? String(newClass) : ''
    }
    console.log('[Correcting] Sending cf_action:', param)
    sendCommand(ws, 'cf_action', param)
}

function finishCorrecting() {
    console.log('[Correcting] User finished correcting')
    interactionState.value = 'none'

    if (status.value.value === 'ask_detecting') {
        if (pendingCFActions.value.length > 0) {
            // 큐잉된 액션을 배치로 전송
            sendCommand(ws, 'cf_batch_commit', {
                robot_name: 'brl_robot',
                actions: pendingCFActions.value
            })
            console.log(`[Correcting] cf_batch_commit sent (${pendingCFActions.value.length} actions)`)
            pendingCFActions.value = []
        } else {
            // 수정 없음 → cf_cancel만
            sendCommand(ws, 'cf_cancel', { robot_name: 'brl_robot' })
            console.log('[Correcting] cf_cancel sent - no changes')
        }

        status.value = status_list.find((e) => e.value === 'picking')
        console.log('[Correcting] ask_detecting -> picking')
    } else if (status.value.value === 'correct_picking') {
        // CF 종료 명령 전송
        sendCommand(ws, 'cf_cancel', { robot_name: 'brl_robot' })
        console.log('[Correcting] cf_cancel sent - CF ended')

        status.value = status_list.find((e) => e.value === 'picking')
        console.log('[Correcting] correct_picking -> picking')
    }
}

function handleHeldTomatoClass(classValue) {
    // UPDATE_SCAN_RESULT 전송 (픽셀 좌표 불필요)
    const param = {
        action: 'UPDATE_SCAN_RESULT',
        pixel_x: 0,
        pixel_y: 0,
        camera: 'camera1',
        robot_name: 'brl_robot',
        new_class: String(classValue)
    }
    console.log('[Correcting] Sending UPDATE_SCAN_RESULT:', param)
    sendCommand(ws, 'cf_action', param)

    // 알림 표시
    $q.notify({
        message: `토마토 상태: ${classValue === 2 ? '썩음' : '안썩음'}`,
        color: 'positive',
        icon: 'edit',
        timeout: 1500
    })

    // cf_cancel 제거 - onRunning()이 자동으로 endCF() 호출하여 BT 재개
    status.value = status_list.find((e) => e.value === 'placing')
    interactionState.value = 'none'
    console.log('[Correcting] correct_placing -> placing')
}

function cancelCorrectPlacing() {
    // CF 취소 & placing 복귀
    sendCommand(ws, 'cf_cancel', { robot_name: 'brl_robot' })
    status.value = status_list.find((e) => e.value === 'placing')
    interactionState.value = 'none'
    console.log('[Correcting] correct_placing cancelled -> placing')
}

function handlePickingClassResponse(classValue) {
    // UPDATE_CLASS 전송
    sendCorrectingAction('UPDATE_CLASS', classValue)

    const classLabels = { 0: '익음', 1: '안익음' }
    $q.notify({
        message: `토마토 상태: ${classLabels[classValue]}`,
        color: 'positive',
        icon: 'edit',
        timeout: 1500
    })

    // cf_cancel 제거 - onRunning()이 자동으로 endCF() 호출하여 BT 재개
    status.value = status_list.find((e) => e.value === 'picking')
    interactionState.value = 'none'
    clickedTomato.value = null
    console.log(`[Correcting] correct_picking -> picking (class=${classValue})`)
}

// ============================================================
// DEV: 테스트용 상태 변경
// ============================================================
function onDevStatusChange(newStatusObj) {
    // 상태 객체 업데이트
    status.value = newStatusObj
    console.log(`[DEV] Status changed to: ${newStatusObj.value}`)

    // 상태에 따른 관련 플래그 업데이트
    if (newStatusObj.value === 'active') {
        is_playing.value = false
        interactionState.value = 'none'
        isReturning.value = false
    } else if (newStatusObj.value === 'halted') {
        is_playing.value = false
        interactionState.value = 'none'
    } else if (newStatusObj.value === 'ask_detecting') {
        is_playing.value = true
        interactionState.value = 'guide'
        pendingCFActions.value = []  // 큐 초기화
        // CF 트리거 전송 (ask_detecting 진입 시 필수)
        // sendCommand(ws, 'cf_trigger', { robot_name: 'brl_robot' })
        // console.log('[DEV] cf_trigger sent for ask_detecting')
    } else if (newStatusObj.value === 'correct_picking') {
        is_playing.value = true
        interactionState.value = 'guide'
        // CF 트리거 전송
        sendCommand(ws, 'cf_trigger', { robot_name: 'brl_robot' })
        console.log('[DEV] cf_trigger sent for correct_picking')
    } else if (newStatusObj.value === 'correct_placing') {
        is_playing.value = true
        interactionState.value = 'select_held_class'
        // CF 트리거 전송
        sendCommand(ws, 'cf_trigger', { robot_name: 'brl_robot' })
        console.log('[DEV] cf_trigger sent for correct_placing')
    } else if (newStatusObj.value === 'correct_proactive') {
        is_playing.value = true
        statusBuffer.value = status_list.find((e) => e.value === 'picking')
        askClassData.value = { suggested_class: 0, pixel_x: 100, pixel_y: 100 }
        interactionState.value = 'ask_class'
        sendCommand(ws, 'cf_trigger', { robot_name: 'brl_robot' })
        console.log('[DEV] cf_trigger sent for correct_proactive')
    } else {
        // preparing_nav, navigating, scanning, picking, placing
        is_playing.value = true
        interactionState.value = 'none'
    }
}

// (제거됨: confirmDelete, confirmAdd)

function closeDialog() {
    dialog.value = null;
    status.value = statusBuffer.value;
}

const showQuitRobotDialog = ref(false)
const quitRobotStatus = ref('ask')

function quitRobot() {
    quitRobotStatus.value = 'quitting'
    sendCommand(ws, 'end_job')
}

function videoClicked(event) {
    const video = event.target
    const rect = video.getBoundingClientRect()
    // ... (좌표 계산 로직 생략, 위쪽 코드와 동일하므로 여기선 내부 로직만 변경) ...
    const displayX = event.clientX - rect.left
    const displayY = event.clientY - rect.top
    const sourceW = video.videoWidth || video.naturalWidth || 640
    const sourceH = video.videoHeight || video.naturalHeight || 480
    const scaleX = sourceW / rect.width
    const scaleY = sourceH / rect.height
    const pixelX = Math.round(displayX * scaleX)
    const pixelY = Math.round(displayY * scaleY)

    // ============================================================
    // 1. ask_detecting 상태 인터랙션 (토마토 수정)
    // ============================================================
    // TODO: 실제 환경에서는 백엔드에서 토마토 detection 데이터를 받아와서
    //       클릭한 위치가 토마토인지 판별해야 함
    // 현재는 시뮬레이션용으로 항상 'ask_add' (토마토 추가 질문) 표시
    // ============================================================
    if (status.value.value === 'ask_detecting') {
        console.log(`[Correcting] Clicked at (${pixelX}, ${pixelY})`)
        lastClickedPixel.value = { x: pixelX, y: pixelY, camera: 'camera1' }

        // 클릭 위치에 토마토 존재 여부 확인
        const existingTomato = findTomatoAtPixel(pixelX, pixelY)

        if (existingTomato) {
            // 토마토 존재 → "여기에 토마토가 없나요?" 확인
            clickedTomato.value = existingTomato
            interactionState.value = 'ask_confirm_delete'
            console.log(`[Correcting] Tomato found at click, asking confirm delete`)
        } else {
            // 토마토 없음 → "여기에 토마토가 있나요?" 확인
            clickedTomato.value = null
            interactionState.value = 'ask_confirm_add'
            console.log(`[Correcting] No tomato at click, asking confirm add`)
        }
        return
    }

    // ============================================================
    // 2. correct_picking 상태: UPDATE_CLASS만 가능
    // ============================================================
    if (status.value.value === 'correct_picking') {
        console.log(`[Correcting] Clicked at (${pixelX}, ${pixelY})`)
        lastClickedPixel.value = { x: pixelX, y: pixelY, camera: 'camera1' }

        // 클릭 위치에 토마토 존재 여부 확인
        const existingTomato = findTomatoAtPixel(pixelX, pixelY)

        if (existingTomato) {
            // 토마토 존재 → 익음/안익음 질문 표시
            clickedTomato.value = existingTomato
            interactionState.value = 'ask_picking_class'
            console.log(`[Correcting] Tomato found, asking class selection`)
        } else {
            $q.notify({
                message: '토마토가 없는 위치입니다. 토마토를 클릭해주세요.',
                color: 'warning',
                icon: 'warning',
                timeout: 1500
            })
        }
        return
    }

    if (status.value.value === 'finding_missed_tomato') {
        dialog.value = 'missed_tomato_true'
        status.value = status_list.find((e) => e.value === 'harvesting')
        setTimeout(() => {
            dialog.value = null
            status.value = status_list.find((e) => e.value === 'harvesting')
        }, 5000)
    }
}

function handleInteractionResponse(response) {
    // 취소/아니오 처리
    if (response === 'no' || response === 'cancel') {
        interactionState.value = 'guide'
        return
    }

    // ask_confirm_delete: 토마토 삭제 확인 (ask_detecting)
    if (response === 'confirm_delete_yes') {
        if (status.value.value === 'ask_detecting') {
            // 큐에 저장 (즉시 전송 안 함)
            pendingCFActions.value.push({
                action: 'DELETE_TOMATO',
                pixel_x: lastClickedPixel.value.x,
                pixel_y: lastClickedPixel.value.y,
                camera: lastClickedPixel.value.camera || 'camera1'
            })
            $q.notify({ message: `삭제 예약됨 (${pendingCFActions.value.length}건)`, color: 'info', icon: 'bookmark', timeout: 1500 })
            console.log(`[Correcting] DELETE_TOMATO queued (total: ${pendingCFActions.value.length})`)
        } else {
            sendCorrectingAction('DELETE_TOMATO')
        }
        clickedTomato.value = null
        interactionState.value = 'guide'
        return
    }

    // ask_confirm_add: 토마토 추가 확인 (ask_detecting, 익음으로 추가)
    if (response === 'confirm_add_yes') {
        if (status.value.value === 'ask_detecting') {
            // 큐에 저장 (즉시 전송 안 함)
            pendingCFActions.value.push({
                action: 'ADD_TOMATO',
                pixel_x: lastClickedPixel.value.x,
                pixel_y: lastClickedPixel.value.y,
                camera: lastClickedPixel.value.camera || 'camera1',
                new_class: 0
            })
            $q.notify({ message: `추가 예약됨 (${pendingCFActions.value.length}건)`, color: 'info', icon: 'bookmark', timeout: 1500 })
            console.log(`[Correcting] ADD_TOMATO queued (total: ${pendingCFActions.value.length})`)
        } else {
            sendCorrectingAction('ADD_TOMATO', 0)
        }
        interactionState.value = 'guide'
        return
    }

    // select_action: 삭제/추가 선택
    if (interactionState.value === 'select_action') {
        if (response === 'delete') {
            sendCorrectingAction('DELETE_TOMATO')
            // Notification handled by setMessageHandler
            interactionState.value = 'guide'
        } else if (response === 'add') {
            interactionState.value = 'select_class'
        }
        return
    }

    // select_class: 익음/안익음 선택
    if (interactionState.value === 'select_class') {
        const classValue = response === 'ripe' ? 0 : 1
        sendCorrectingAction('ADD_TOMATO', classValue)
        // Notification handled by setMessageHandler
        interactionState.value = 'guide'
        return
    }

    // ask_rotten: 썩음 여부 (기존 로직 유지)
    if (response === 'rotten_yes') {
        // 썩은 토마토 - placing 후 active로 복귀
        status.value = status_list.find((e) => e.value === 'placing')
        interactionState.value = 'none'
        setTimeout(() => {
            status.value = status_list.find((e) => e.value === 'active')
            is_playing.value = false
        }, 3000)
    } else if (response === 'rotten_no') {
        status.value = status_list.find((e) => e.value === 'placing')
        interactionState.value = 'none'
        setTimeout(() => {
            status.value = status_list.find((e) => e.value === 'active')
            is_playing.value = false
        }, 3000)
    }
}

function handleAskClassResponse(selectedClass) {
    // 타임아웃 클리어
    if (askClassTimeout.value) {
        clearTimeout(askClassTimeout.value)
        askClassTimeout.value = null
    }

    // 백엔드에 응답 전송
    sendCommand(ws, 'cf_ask_class_response', { new_class: selectedClass })

    const classLabels = { 0: '익음', 1: '안익음', 2: '썩음' }
    $q.notify({
        message: `토마토 상태: ${classLabels[selectedClass] || selectedClass}`,
        color: 'positive',
        icon: 'check_circle',
        timeout: 1500
    })

    // 상태 초기화: correct_* 패턴으로 이전 상태 복원
    askClassData.value = null
    interactionState.value = 'none'
    if (statusBuffer.value) {
        status.value = statusBuffer.value
        statusBuffer.value = null
    }
    console.log(`[AskClass] Response sent: class=${selectedClass}, status restored`)
}

// function changeStatus(status) {
//     console.log(status)
//     ws.send(JSON.stringify({command: "change_status", param: status}))
// }

onMounted(() => {
    // Use setMessageHandler for reconnect-safe message handling
    setMessageHandler((event) => {
        const msg = JSON.parse(event.data)

        if (msg.type === 'ask_fact') {
            console.log('[WS] ask_fact received:', msg.value)
            clearFactQuery()
            pendingFactQuery.value = msg.value
            factQueryVisible.value = true

            const timeoutSec = Number(msg.value.timeout_sec || 30)
            factQueryTimeout.value = setTimeout(() => {
                if (pendingFactQuery.value?.request_id === msg.value.request_id) {
                    clearFactQuery()
                    $q.notify({
                        message: '질문 응답 시간이 만료되었습니다.',
                        color: 'warning',
                        icon: 'schedule'
                    })
                }
            }, timeoutSec * 1000)
            return
        }

        // Scene Graph 데이터 저장
        if (msg.type === 'scene_graph') {
            sceneGraphData.value = msg.value
            console.log('[WS] Scene graph updated:', msg.value)
            return
        }

        // CF 응답 처리
        if (msg.type === 'cf_response') {
            console.log('[WS] CF Response:', msg)
            if (msg.success) {
                let notifMsg = `${msg.action} 성공`
                let notifIcon = 'check_circle'
                
                if (msg.action === 'ADD_TOMATO') {
                    notifMsg = '토마토가 정상적으로 추가되었습니다.'
                    notifIcon = 'add_circle'
                } else if (msg.action === 'DELETE_TOMATO') {
                    notifMsg = '토마토가 삭제되었습니다.'
                    notifIcon = 'delete'
                } else if (msg.action === 'UPDATE_CLASS') {
                    notifMsg = '토마토 상태가 수정되었습니다.'
                    notifIcon = 'edit'
                }

                $q.notify({
                    message: notifMsg,
                    color: 'positive',
                    icon: notifIcon,
                    timeout: 2000
                })
            } else {
                $q.notify({
                    message: `요청 실패: ${msg.message}`,
                    color: 'negative',
                    icon: 'error'
                })
            }
            return
        }

        // ask_detection 처리 (KB에서 감지 완료 시 확인 요청)
        if (msg.type === 'ask_detection') {
            console.log('[WS] ask_detection received:', msg.value)

            // 상태를 ask_detecting으로 변경하고 사용자 가이드 UI 표시
            statusBuffer.value = status.value
            status.value = status_list.find((e) => e.value === 'ask_detecting')
            
            pendingCFActions.value = [] // 큐 초기화
            interactionState.value = 'guide'
            return
        }

        // ask_class 처리 (KB에서 confidence 낮은 토마토 확인 요청)
        if (msg.type === 'ask_class') {
            console.log('[WS] ask_class received:', msg.value)

            // correct_* 패턴: 이전 상태 저장 후 전환
            statusBuffer.value = status.value
            status.value = status_list.find((e) => e.value === 'correct_proactive')

            askClassData.value = msg.value
            interactionState.value = 'ask_class'

            // 12초 후 자동 제출 (suggested_class 사용)
            if (askClassTimeout.value) clearTimeout(askClassTimeout.value)
            askClassTimeout.value = setTimeout(() => {
                if (interactionState.value === 'ask_class' && askClassData.value) {
                    console.log('[AskClass] Timeout - auto-submitting suggested class')
                    handleAskClassResponse(askClassData.value.suggested_class)
                }
            }, 12000)
            return
        }

        // CF 배치 결과 처리
        if (msg.type === 'cf_batch_result') {
            console.log('[WS] CF Batch Result:', msg)
            if (msg.success) {
                $q.notify({
                    message: `${msg.total}건 수정 완료`,
                    color: 'positive',
                    icon: 'check_circle',
                    timeout: 2000
                })
            } else {
                const failCount = msg.results.filter(r => !r.success).length
                $q.notify({
                    message: `${msg.total}건 중 ${failCount}건 실패`,
                    color: 'warning',
                    icon: 'warning',
                    timeout: 3000
                })
            }
            return
        }

        // 기존 data 처리 (CF 관련 상태에서는 백엔드 데이터 무시)
        if (msg.type === 'data') {
            console.log('[WS] Data received:', msg.value)
        }
        if (msg.type === 'data' && !['correct_placing', 'correct_picking', 'ask_detecting', 'correct_proactive'].includes(status.value.value)) {
            const data = JSON.parse(msg.value)
            console.log('[WS] Parsed status:', data.status, 'playing:', data.playing)

            // 백엔드에서 결정된 상태를 그대로 사용
            const found = status_list.find((e) => e.value === data.status)
            if (found) status.value = found
            is_playing.value = Boolean(data.playing)

            // 종료 완료 감지: bt_job_done 수신 시 다이얼로그 업데이트 후 홈 이동
            if (data.status === 'bt_job_done' && quitRobotStatus.value === 'quitting') {
                quitRobotStatus.value = 'quitted'
                console.log('[WS] bt_job_done received - robot returned to base')
                setTimeout(() => {
                    $router.push('/')
                }, 3000)
            }
        }
    })
})


</script>

<style scoped>
.ask-class-bbox {
    position: absolute;
    border: 3px solid #FF4444;
    border-radius: 6px;
    pointer-events: none;
    z-index: 5;
    animation: bbox-pulse 1.2s ease-in-out infinite;
}

@keyframes bbox-pulse {
    0%, 100% { border-color: #FF4444; box-shadow: 0 0 8px rgba(255, 68, 68, 0.6); }
    50% { border-color: #FF8888; box-shadow: 0 0 16px rgba(255, 68, 68, 0.9); }
}
</style>
