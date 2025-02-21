# KIPRIS & Gemini 특허 분석 시스템

## 프로젝트 소개

이 Streamlit 애플리케이션은 KIPRIS API와 Google Gemini AI를 활용하여 특허 데이터를 종합적으로 분석하고 시각화하는 도구입니다. 키워드 또는 발명자를 기반으로 특허를 검색하고, 다양한 관점에서 심층 분석을 제공합니다.

## 주요 기능

- KIPRIS API를 통한 특허 검색
- Google Gemini AI를 활용한 특허 데이터 인사이트 분석
- 다양한 시각화 및 통계 제공:
  - 연도별 특허 출원 동향
  - 출원인 및 발명자 분석
  - 기술 분야(IPC) 분포
  - 키워드 트렌드 분석
  - 기술 성숙도 평가
  - 협업 네트워크 시각화

## 사전 준비사항

1. Python 3.8 이상
2. KIPRIS API 키
3. Google Gemini API 키

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/your-username/patent-analysis-app.git
cd patent-analysis-app
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows의 경우 `venv\Scripts\activate`
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

4. Streamlit 시크릿 설정:
`secrets.toml` 파일을 생성하고 다음 형식으로 API 키 추가:
```toml
KIPRIS_API_KEY = "your_kipris_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
```

## 실행 방법

```bash
streamlit run new_app.py
```

## 환경 설정

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- NetworkX
- Google Generative AI

## 주요 기술 스택

- 데이터 처리: Pandas
- 데이터 시각화: Plotly
- 웹 애플리케이션: Streamlit
- AI 분석: Google Gemini
- 특허 데이터: KIPRIS API

## 라이선스

[적절한 라이선스 추가 - MIT, Apache 등]

## 기여 방법

1. 포크(Fork)
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시 (`git push origin feature/AmazingFeature`)
5. 풀 리퀘스트 오픈

## 문의 및 지원

문제나 제안사항이 있으시면 [이슈 트래커](https://github.com/your-username/patent-analysis-app/issues)를 통해 알려주세요.

## 스크린샷

[애플리케이션 주요 화면 스크린샷 추가 예정]

**면책조항:**
- 본 애플리케이션은 교육 및 연구 목적으로 제작되었습니다.
- 특허 데이터의 정확성과 완전성을 보장하지 않습니다.
- KIPRIS와 Google의 서비스 약관을 준수해야 합니다.



# 가상환경 생성
python -m venv kipris_env

# 가상환경 활성화 (Windows)
kipris_env\Scripts\activate

# 필요한 패키지 설치
pip install streamlit pandas plotly requests xmltodict

# streamlit 실행
streamlit run app.py

#패키지 설치
python -m pip install streamlit pandas plotly requests xmltodict google.generativeai

