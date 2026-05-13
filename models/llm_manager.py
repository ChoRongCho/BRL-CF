# llm_manager.py
"""
LLM Manager - LLM 연동 싱글톤 클래스

기능:
- 모델 캐싱 및 재사용
- 토큰 사용량 추적
- LangSmith tracing 지원
- Local VLLM (Qwen) 서버 연동
"""

import os
from importlib import import_module
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_community.callbacks import get_openai_callback

load_dotenv()


class LLMManager:
    """
    BRL 시스템 통합 LLM Manager (Singleton)

    주요 기능:
    - 중앙집중식 모델 설정
    - 환경 변수 기반 API 키 관리
    - 모델 캐싱 및 재사용
    - 토큰 사용량 및 비용 추적
    """
    
    _instance = None
    _model_cache: Dict[str, ChatOpenAI] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._load_environment()
            self._reset_metrics()
            self._initialized = True
    
    def _load_environment(self):
        """환경 변수 로드 (API 키, LangSmith tracing, Qwen 서버 설정)"""
        # 로컬 서버는 실제 키가 필요 없으므로 'dummy'를 기본값으로 사용 (아무 값이나 가능)
        self.api_key = os.getenv("OPENAI_API_KEY", "dummy")
        
        # TE805A VLM 서버 주소 설정
        self.base_url = os.getenv("OPENAI_BASE_URL", "http://163.239.227.136:8000/v1")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
        self.langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
        self.langchain_apikey = os.getenv("LANGCHAIN_API_KEY")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT")

        # langchain_tracing 정보 출력
        print(f"=== LangSmith Debug ===")
        print(f"Tracing Enabled: {self.langchain_tracing}")
        print(f"Project Name: {self.langchain_project}")
        
        
    def _reset_metrics(self):
        """사용량 메트릭 초기화"""
        self.metrics = {
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_calls": 0,
            "calls_by_step": {},
            "tokens_by_step": {},
            "cost_by_step": {}
        }
    
    def get_model(self,
                #   model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct", # [변경] 기본 모델 Qwen으로 변경
                  model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct", # [변경] Qwen3.5로 업데이트
                  temperature: float = 0.0,
                  enable_thinking: bool = False,
                  stop: Optional[list] = None,
                  max_tokens: Optional[int] = None) -> ChatOpenAI:
        """
        설정된 LLM 모델 인스턴스 반환 (캐싱 적용)

        Args:
            model_name: OpenAI 모델 이름
            temperature: 샘플링 온도 (0.0 = 결정적 출력)
            enable_thinking: Thinking 모드 활성화 여부 (추론 과정에서 중간 생각 출력)
            stop: 중단 시퀀스
            max_tokens: 최대 생성 토큰 수

        Returns:
            ChatOpenAI: 설정된 모델 인스턴스
        """
        # 캐시 키 생성 (동일 설정 시 재사용)

        cache_key = f"{model_name}_{temperature}_{enable_thinking}_{stop}_{max_tokens}"
        
        if cache_key not in self._model_cache:
            model_config = {
                "model": model_name,
                "temperature": temperature,
                "api_key": self.api_key,
                "base_url": self.base_url, # vlm 서버 주소
                # Qwen(OpenAI-compatible) 서버에서 thinking 모드 on/off를 실제 요청 바디에 반영
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking
                    }
                },
            }
            
            if stop is not None:
                model_config["stop"] = stop
            if max_tokens is not None:
                model_config["max_tokens"] = max_tokens
                
            self._model_cache[cache_key] = ChatOpenAI(**model_config)
        
        return self._model_cache[cache_key]
    
    def record_llm_usage(self, step_name: str, tokens_input: int, tokens_output: int, cost: float = 0.0):
        """
        LLM 사용량 메트릭 기록

        Args:
            step_name: 파이프라인 단계 이름 (예: "brl_preprocess_reasoning")
            tokens_input: 입력 토큰 수
            tokens_output: 출력 토큰 수
            cost: 예상 비용 (USD)
        """
        # 전체 합계 업데이트
        self.metrics["total_tokens_input"] += tokens_input
        self.metrics["total_tokens_output"] += tokens_output
        self.metrics["total_tokens"] += tokens_input + tokens_output
        self.metrics["total_cost"] += cost
        self.metrics["total_calls"] += 1
        
        # 단계별 메트릭 업데이트
        if step_name not in self.metrics["calls_by_step"]:
            self.metrics["calls_by_step"][step_name] = 0
            self.metrics["tokens_by_step"][step_name] = {"input": 0, "output": 0, "total": 0}
            self.metrics["cost_by_step"][step_name] = 0.0
        
        self.metrics["calls_by_step"][step_name] += 1
        self.metrics["tokens_by_step"][step_name]["input"] += tokens_input
        self.metrics["tokens_by_step"][step_name]["output"] += tokens_output
        self.metrics["tokens_by_step"][step_name]["total"] += tokens_input + tokens_output
        self.metrics["cost_by_step"][step_name] += cost
    
    def invoke_with_callback(self, chain: Runnable, input_data: Dict[str, Any], step_name: str) -> Tuple[Any, Any]:
        """
        LLM 체인 실행 및 자동 토큰 사용량 추적

        Args:
            chain: 실행할 LangChain runnable chain
            input_data: 체인 입력 데이터
            step_name: 메트릭 추적용 단계 이름

        Returns:
            (체인 실행 결과, callback 정보)
        """
        with get_openai_callback() as cb:
            result = chain.invoke(input_data)
        
        # 실제 토큰 사용량 자동 기록
        self.record_llm_usage(
            step_name=step_name,
            tokens_input=cb.prompt_tokens,
            tokens_output=cb.completion_tokens,
            cost=cb.total_cost
        )
        
        return result, cb

    def make_refining_query(self, domain_name: str, target_fact: str, action_name: Optional[str] = None) -> str:
        """Convert a symbolic fact query into a natural-language yes/no question."""
        prompt = self._load_domain_prompt(domain_name)
        messages = prompt.refining_query_prompt(target_fact, action_name)
        model = self.get_model(temperature=0.0, max_tokens=80)
        response = model.invoke(messages)
        return response.content.strip()

    def _load_domain_prompt(self, domain_name: str):
        module = import_module(f"models.{domain_name}.prompt")
        return module.Prompt()
    
    def get_metrics(self) -> Dict[str, Any]:
        """현재 LLM 사용량 메트릭 반환"""
        return self.metrics.copy()

    def reset_metrics(self):
        """메트릭 초기화 (새 평가 시작 시)"""
        self._reset_metrics()

    def clear_cache(self):
        """모델 캐시 초기화 (메모리 관리용)"""
        self._model_cache.clear()

    def get_environment_info(self) -> Dict[str, Any]:
        """현재 환경 설정 정보 반환 (디버깅용)"""
        return {
            "api_key_configured": bool(self.api_key),
            "base_url": self.base_url,
            "langchain_tracing": self.langchain_tracing,
            "langchain_project": self.langchain_project,
            "cached_models": list(self._model_cache.keys()),
            "current_metrics": self.metrics
        }

# 전역 인스턴스 (지연 초기화)
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """전역 LLM Manager 인스턴스 반환 (싱글톤)"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager
