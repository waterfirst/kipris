import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # 추가
from datetime import datetime
import requests
import xmltodict
import google.generativeai as genai
from typing import Dict, List
import networkx as nx  # 추가
from collections import Counter
import re

# 스타일 설정
COLORS = {
    "primary": "#1E88E5",
    "secondary": "#FFC107",
    "background": "#0E1117",
    "text": "#FFFFFF",
    "graph": ["#1E88E5", "#FFC107", "#4CAF50", "#E91E63", "#9C27B0"],
}
ipc_descriptions = {
    'A': '생활필수품',
    'B': '처리조작/운수',
    'C': '화학/야금',
    'D': '섬유/지류',
    'E': '고정구조물',
    'F': '기계공학',
    'G': '물리학',
    'H': '전기',
}
# Streamlit 페이지 설정
st.set_page_config(page_title="KIPRIS & Gemini 특허 분석 시스템", layout="wide")

# API 키 로드
try:
    KIPRIS_API_KEY = st.secrets["KIPRIS_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error("secrets.toml 파일에서 API 키를 로드할 수 없습니다.")
    KIPRIS_API_KEY = None
    GEMINI_API_KEY = None

class KiprisAPI:
    def __init__(self, api_key: str):
        """KIPRIS API 클라이언트를 초기화합니다."""
        self.api_key = api_key
        self.base_url = "http://plus.kipris.or.kr/kipo-api/kipi"

    def search_patents(self, search_type: str, search_query: str, min_results: int = 50) -> List[Dict]:
        """특허 검색을 수행합니다. 최소 min_results개의 결과를 반환하려 시도합니다."""
        results = []
        page = 1
        max_pages = 10  # 최대 10페이지까지 검색 (5000개 결과 가능)

        while len(results) < min_results and page <= max_pages:
            endpoint = "/patUtiModInfoSearchSevice/getWordSearch"  # API 문서에 따른 엔드포인트
            params = {
                "ServiceKey": self.api_key,
                "pageNo": str(page),
                "numOfRows": "50",  # 한 페이지당 최대 결과 수
                "year": "10",       # 최근 10년 데이터
                "patent": "true",   # 특허 포함
                "utility": "true"   # 실용 포함
            }
            
            if search_type == "키워드":
                params["word"] = search_query
            else:  # 대표발명자 검색
                st.warning("현재 API는 발명자 검색을 지원하지 않습니다. 키워드 검색으로 대체합니다.")
                params["word"] = search_query  # 임시로 키워드 검색으로 처리

            try:
                response = self._make_request(endpoint, params)
                page_results = self._parse_search_response(response)
                if not page_results:
                    break
                results.extend(page_results)
                page += 1
            except Exception as e:
                st.error(f"검색 중 오류 발생 (페이지 {page}): {str(e)}")
                break

        return results[:min_results] if len(results) > min_results else results

    def _make_request(self, endpoint: str, params: Dict) -> requests.Response:
        """API 요청을 수행합니다."""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"API 호출 실패 (상태 코드: {response.status_code})")
        return response

    def _parse_search_response(self, response: requests.Response) -> List[Dict]:
        """검색 결과를 파싱합니다."""
        try:
            dict_data = xmltodict.parse(response.content)
            items = (
                dict_data.get("response", {})
                .get("body", {})
                .get("items", {})
                .get("item", [])
            )
            if not items:
                return []
            if isinstance(items, dict):
                items = [items]
            return [
                {
                    "applicationNumber": item.get("applicationNumber", ""),
                    "applicantName": item.get("applicantName", ""),
                    "inventionTitle": item.get("inventionTitle", ""),
                    "astrtCont": item.get("astrtCont", ""),
                    "openDate": item.get("openDate", ""),
                    "openNumber": item.get("openNumber", ""),
                    "publicationDate": item.get("publicationDate", ""),
                    "publicationNumber": item.get("publicationNumber", ""),
                    "registerDate": item.get("registerDate", ""),
                    "registerNumber": item.get("registerNumber", ""),
                    "registerStatus": item.get("registerStatus", ""),
                    "applicationDate": item.get("applicationDate", ""),
                    "inventorName": item.get("inventorName", ""),  # API에서 제공되지 않을 수 있음
                    "ipcNumber": item.get("ipcNumber", "")          # IPC 코드 추가
                }
                for item in items
            ]
        except Exception as e:
            st.error(f"응답 파싱 중 오류 발생: {str(e)}")
            return []
@st.cache_data
def analyze_patents_with_gemini(patents: List[Dict], api_key: str) -> str:
    """Gemini API를 사용하여 특허 데이터를 분석합니다."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")

    # 분석을 위한 특허 데이터 준비 (최대 50개 특허 분석)
    patent_summaries = "\n".join(
        [
            f"""특허 {idx+1}:
제목: {p['inventionTitle']}
요약: {p['astrtCont']}
출원인: {p['applicantName']}
대표발명자: {p['inventorName']}
출원번호: {p['applicationNumber']}
출원일자: {p['applicationDate']}
공개일자: {p['openDate']}
등록상태: {p['registerStatus']}
등록일자: {p['registerDate']}\n"""
            for idx, p in enumerate(patents)
        ]
    )

    prompt = f"""
    다음 {len(patents)}개의 특허 데이터를 종합적으로 분석하여 주요 트렌드와 인사이트를 도출해주세요:
    
    {patent_summaries}
    
    다음 항목들을 포함해 상세히 분석해주세요:
    1. 주요 기술 분야 및 트렌드
    2. 주요 출원인 및 발명자 분석
    3. 기술적 특징 및 혁신 포인트
    4. 특허의 법적 상태 분석
    5. 시계열적 기술 발전 방향
    6. 산업적 응용 가능성 및 시장 영향
    7. 기술 분야별 특허 집중도
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API 분석 중 오류 발생: {str(e)}")
        return "분석 실패"
def analyze_yearly_keywords(df: pd.DataFrame) -> dict:
    """연도별 주요 키워드를 추출하고 분석합니다."""
    from collections import Counter
    import re
    
    # 불용어 정의
    stopwords = set(['및', '이를', '있는', '위한', '하는', '또는', '되는', '통한', '있다', '한다', 
                    '본', '발명', '기술', '장치', '있을', '수행', '형성', '구성'])
    
    yearly_keywords = {}
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        
        # 제목과 요약에서 키워드 추출
        text = ' '.join(year_df['inventionTitle'].fillna('') + ' ' + year_df['astrtCont'].fillna(''))
        
        # 단어 추출 (2글자 이상)
        words = [word for word in re.findall(r'\w+', text) 
                if len(word) >= 2 and word not in stopwords]
        
        # 빈도수 계산
        word_counts = Counter(words)
        
        # 상위 10개 키워드 저장
        yearly_keywords[year] = {
            'keywords': [word for word, _ in word_counts.most_common(10)],
            'counts': [count for _, count in word_counts.most_common(10)]
        }
    
    return yearly_keywords
def create_keyword_trend_visualization(yearly_keywords: dict) -> go.Figure:
    """연도별 키워드 트렌드를 시각화합니다."""
    fig = go.Figure()
    
    # px.colors.qualitative.Set3 대신 수동으로 컬러 팔레트 정의
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 전체 기간 동안의 상위 키워드 추출
    all_keywords = set()
    for year_data in yearly_keywords.values():
        all_keywords.update(year_data['keywords'][:5])  # 상위 5개만 사용
    
    # 각 키워드별 연도별 빈도 추적
    for idx, keyword in enumerate(all_keywords):
        years = []
        counts = []
        for year in sorted(yearly_keywords.keys()):
            year_data = yearly_keywords[year]
            try:
                idx_in_year = year_data['keywords'].index(keyword)
                count = year_data['counts'][idx_in_year]
            except ValueError:
                count = 0
            years.append(year)
            counts.append(count)
            
        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            name=keyword,
            mode='lines+markers',
            line=dict(color=colors[idx % len(colors)]),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="연도별 주요 키워드 트렌드",
        xaxis_title="연도",
        yaxis_title="키워드 출현 빈도",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
    )
    
    return fig
def create_keyword_trend_visualization(yearly_keywords: dict) -> go.Figure:
    """연도별 키워드 트렌드를 시각화합니다."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    # 전체 기간 동안의 상위 키워드 추출
    all_keywords = set()
    for year_data in yearly_keywords.values():
        all_keywords.update(year_data['keywords'][:5])  # 상위 5개만 사용
    
    # 각 키워드별 연도별 빈도 추적
    for idx, keyword in enumerate(all_keywords):
        years = []
        counts = []
        for year in sorted(yearly_keywords.keys()):
            year_data = yearly_keywords[year]
            try:
                idx_in_year = year_data['keywords'].index(keyword)
                count = year_data['counts'][idx_in_year]
            except ValueError:
                count = 0
            years.append(year)
            counts.append(count)
            
        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            name=keyword,
            mode='lines+markers',
            line=dict(color=colors[idx % len(colors)]),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="연도별 주요 키워드 트렌드",
        xaxis_title="연도",
        yaxis_title="키워드 출현 빈도",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
    )
    
    return fig

def analyze_ipc_codes(df: pd.DataFrame) -> pd.DataFrame:
    """IPC 코드를 분석하여 기술 분야별 분포를 파악합니다."""
    import re
    
    # IPC 코드 분리 및 정규화
    ipc_sections = []
    for ipc in df['ipcNumber'].fillna('').str.split('|'):
        sections = [code.strip()[:4] for code in ipc if code.strip()]
        ipc_sections.extend(sections)
    
    ipc_counts = pd.Series(ipc_sections).value_counts()
    
    # IPC 코드 설명 매핑
    ipc_descriptions = {
        'A': '생활필수품',
        'B': '처리조작/운수',
        'C': '화학/야금',
        'D': '섬유/지류',
        'E': '고정구조물',
        'F': '기계공학',
        'G': '물리학',
        'H': '전기',
    }
    
    return pd.DataFrame({
        'ipc': ipc_counts.index,
        'count': ipc_counts.values,
        'description': [f"{code[0]}: {ipc_descriptions.get(code[0], '기타')}" for code in ipc_counts.index],
        'section': [code[0] for code in ipc_counts.index]
    })
def calculate_avg_registration_period(df: pd.DataFrame) -> float:
    """출원일자와 등록일자 간의 평균 기간(개월 단위)을 계산합니다."""
    # 날짜 형식이 올바르지 않을 수 있으므로 안전하게 변환
    df['applicationDate'] = pd.to_datetime(df['applicationDate'], errors='coerce')
    df['registerDate'] = pd.to_datetime(df['registerDate'], errors='coerce')
    
    # 등록일자가 있는 데이터만 필터링
    valid_df = df.dropna(subset=['applicationDate', 'registerDate'])
    
    if valid_df.empty:
        return 0.0  # 유효한 데이터가 없으면 0 반환
    
    # 기간 계산 (일 단위)
    days_diff = (valid_df['registerDate'] - valid_df['applicationDate']).dt.days  # 수정: dt(days) -> dt.days
    
    # 평균 계산 후 월 단위로 변환 (1개월 ≈ 30.44일)
    avg_days = days_diff.mean()
    avg_months = avg_days / 30.44
    
    return avg_months
def create_ipc_visualization(ipc_analysis: pd.DataFrame) -> go.Figure:
    """IPC 분석 결과를 시각화합니다."""
    fig = go.Figure()
    
    # Section별로 그룹화하여 시각화
    for section in sorted(ipc_analysis['section'].unique()):
        section_data = ipc_analysis[ipc_analysis['section'] == section]
        fig.add_trace(go.Bar(
            name=ipc_descriptions.get(section, '기타'),
            x=section_data['ipc'],
            y=section_data['count'],
            text=section_data['description'],
            hovertemplate=
            'IPC: %{x}<br>'+
            '건수: %{y}<br>'+
            '분야: %{text}<br>'+
            '<extra></extra>'
        ))
    
    fig.update_layout(
        title="기술 분야(IPC) 분포",
        xaxis_title="IPC 코드",
        yaxis_title="특허 건수",
        barmode='group',
        showlegend=True,
        legend_title="기술 분야",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
    )
    
    return fig

def analyze_technology_maturity(df: pd.DataFrame) -> dict:
    """기술 성숙도를 분석합니다."""
    # 연도별 특허 출원 증가율
    yearly_counts = df.groupby('year').size()
    growth_rates = yearly_counts.pct_change()
    
    maturity_scores = {
        year: {
            'growth_rate': rate,
            'patent_count': count,
            'maturity_stage': 'emerging' if rate > 0.2 else 'mature' if rate > 0 else 'declining'
        }
        for year, rate, count in zip(growth_rates.index, growth_rates.values, yearly_counts.values)
    }
    
    return maturity_scores

def create_maturity_visualization(maturity_data: dict) -> go.Figure:
    """기술 성숙도 분석 결과를 시각화합니다."""
    years = list(maturity_data.keys())
    growth_rates = [data['growth_rate'] for data in maturity_data.values()]
    counts = [data['patent_count'] for data in maturity_data.values()]
    stages = [data['maturity_stage'] for data in maturity_data.values()]
    
    # 성숙도 단계별 색상 매핑
    stage_colors = {
        'emerging': '#4CAF50',
        'mature': '#FFC107',
        'declining': '#F44336'
    }
    
    fig = go.Figure()
    
    # 특허 수 선 그래프
    fig.add_trace(go.Scatter(
        x=years,
        y=counts,
        name='특허 수',
        yaxis='y1',
        line=dict(color='#2196F3', width=2)
    ))
    
    # 성장률 막대 그래프
    fig.add_trace(go.Bar(
        x=years,
        y=growth_rates,
        name='성장률',
        yaxis='y2',
        marker_color=[stage_colors[stage] for stage in stages]
    ))
    
    fig.update_layout(
        title='기술 성숙도 분석',
        yaxis=dict(title='특허 수'),
        yaxis2=dict(title='성장률', overlaying='y', side='right'),
        showlegend=True,
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
    )
    
    return fig

def analyze_collaboration_network(df: pd.DataFrame, column: str) -> go.Figure:
    """출원인 또는 발명자의 협업 네트워크를 분석합니다."""
    from itertools import combinations
    import networkx as nx
    
    # 협업 관계 추출
    collaborations = {}
    for _, group in df.groupby('applicationNumber'):
        names = group[column].unique()
        if len(names) > 1:
            for name1, name2 in combinations(names, 2):
                key = tuple(sorted([name1, name2]))
                collaborations[key] = collaborations.get(key, 0) + 1
    
    # 네트워크 그래프 생성
    G = nx.Graph()
    for (name1, name2), weight in collaborations.items():
        G.add_edge(name1, name2, weight=weight)
    
    # 협업 관계가 없으면 빈 Figure 반환
    if not G.nodes():
        st.warning(f"{column}에 대한 협업 관계가 없습니다.")
        return go.Figure()
    
    # 노드 크기 계산 (특허 수 기반)
    node_sizes = {name: len(df[df[column] == name]) for name in G.nodes()}
    
    # 노드 위치 계산
    pos = nx.spring_layout(G)
    
    # 시각화 생성
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=[node_sizes[node] * 2 for node in G.nodes()],
            color=[len(G[node]) for node in G.nodes()],
            line=dict(width=2)
        ),
        text=[f"{node}\n(특허 수: {node_sizes[node]})" for node in G.nodes()],
        textposition='top center'
    )

    # 엣지와 노드 데이터 추가
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)

    # 최종 시각화
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"{'출원인' if column=='applicantName' else '발명자'} 협업 네트워크",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["background"],
            font=dict(color=COLORS["text"]),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig  # 명시적 반환 추가
@st.cache_data
def analyze_patent_trends(df: pd.DataFrame) -> dict:
    """특허 데이터의 트렌드를 분석합니다."""
    # 연도별 출원 동향
    df["year"] = pd.to_datetime(df["applicationDate"]).dt.year
    yearly_patents = df.groupby("year").size().reset_index(name="count")

    # 출원인별 특허 수
    applicant_patents = df["applicantName"].value_counts().reset_index()
    applicant_patents.columns = ["applicant", "count"]

    # 대표발명자별 특허 수
    inventor_patents = df["inventorName"].value_counts().reset_index()
    inventor_patents.columns = ["inventor", "count"]

    # 등록 상태별 분류
    status_counts = df["registerStatus"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    return {
        "yearly_trend": yearly_patents,
        "applicant_trend": applicant_patents,
        "inventor_trend": inventor_patents,
        "status_counts": status_counts
    }
def filter_patents(df, status_filter, year_range, applicant_filter):
    """특허 데이터를 필터링합니다."""
    filtered_df = df.copy()
    
    # 등록상태 필터
    if status_filter:
        filtered_df = filtered_df[filtered_df['registerStatus'].isin(status_filter)]
    
    # 연도 범위 필터
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]
    
    # 출원인 필터
    if applicant_filter:
        filtered_df = filtered_df[filtered_df['applicantName'].isin(applicant_filter)]
    
    return filtered_df
def create_visualizations(analysis_data: dict) -> dict:
    """분석 데이터를 시각화합니다."""
    # 연도별 트렌드 그래프
    fig_yearly = go.Figure()
    fig_yearly.add_trace(
        go.Scatter(
            x=analysis_data["yearly_trend"]["year"],
            y=analysis_data["yearly_trend"]["count"],
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8),
        )
    )
    fig_yearly.update_layout(
        title="연도별 특허 출원 동향",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )

    # 출원인별 특허 수 그래프
    fig_applicant = go.Figure()
    fig_applicant.add_trace(
        go.Bar(
            x=analysis_data["applicant_trend"]["applicant"][:10],
            y=analysis_data["applicant_trend"]["count"][:10],
            marker_color=COLORS["secondary"],
        )
    )
    fig_applicant.update_layout(
        title="상위 10개 출원인별 특허 수",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )

    # 대표발명자별 특허 수 그래프
    fig_inventor = go.Figure()
    fig_inventor.add_trace(
        go.Bar(
            x=analysis_data["inventor_trend"]["inventor"][:10],
            y=analysis_data["inventor_trend"]["count"][:10],
            marker_color=COLORS["graph"][2],
        )
    )
    fig_inventor.update_layout(
        title="상위 10개 대표발명자별 특허 수",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )

    # 등록 상태별 분포 그래프
    fig_status = go.Figure()
    fig_status.add_trace(
        go.Pie(
            labels=analysis_data["status_counts"]["status"],
            values=analysis_data["status_counts"]["count"],
            marker_colors=COLORS["graph"],
        )
    )
    fig_status.update_layout(
        title="특허 등록 상태 분포",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
    )

    return {
        "yearly_trend": fig_yearly,
        "applicant_trend": fig_applicant,
        "inventor_trend": fig_inventor,
        "status_trend": fig_status
    }
def main():
    st.title("KIPRIS & Gemini 특허 분석 시스템")

    if not KIPRIS_API_KEY or not GEMINI_API_KEY:
        st.error("API 키가 설정되지 않았습니다. secrets.toml 파일을 확인해주세요.")
        st.info("secrets.toml 설정 예시:\n```toml\nKIPRIS_API_KEY = 'your_kipris_api_key'\nGEMINI_API_KEY = 'your_gemini_api_key'\n```")
        return

    with st.sidebar:
        st.header("검색 설정")
        search_type = st.selectbox("검색 유형", ["키워드", "대표발명자"])
        search_query = st.text_input(f"{search_type}를 입력하세요")
        min_results = st.number_input("검색할 특허 수", min_value=25, max_value=50, value=50)
        advanced_analysis = st.checkbox("심층 분석 활성화", value=True)

    if st.sidebar.button("검색 및 분석") and search_query:
        client = KiprisAPI(KIPRIS_API_KEY)

        with st.spinner(f"특허 검색 및 분석 중... (최소 {min_results}개 결과)"):
            patents = client.search_patents(search_type, search_query, min_results)

            if not patents:
                st.warning("검색 결과가 없습니다.")
                return  # 검색 결과가 없으면 종료
            elif len(patents) < min_results:
                st.warning(f"요청하신 {min_results}개보다 적은 {len(patents)}개의 결과만 찾았습니다.")

            df = pd.DataFrame(patents)  # df 정의

            # 기본 분석 및 시각화
            analysis_data = analyze_patent_trends(df)
            visuals = create_visualizations(analysis_data)

            # 심층 분석
            if advanced_analysis:
                df['year'] = pd.to_datetime(df['applicationDate']).dt.year
                yearly_keywords = analyze_yearly_keywords(df)
                keyword_trend_fig = create_keyword_trend_visualization(yearly_keywords)
                ipc_analysis = analyze_ipc_codes(df)
                ipc_fig = create_ipc_visualization(ipc_analysis)
                maturity_analysis = analyze_technology_maturity(df)
                maturity_fig = create_maturity_visualization(maturity_analysis)

            ai_analysis = analyze_patents_with_gemini(patents, GEMINI_API_KEY)

            # 결과 표시
            st.header("특허 분석 결과")
            st.write(f"총 {len(patents)}개의 특허를 분석했습니다.")

            tab1, tab2, tab3, tab4 = st.tabs(["기본 통계", "심층 분석", "AI 분석 리포트", "데이터 테이블"])

            with tab1:
                st.subheader("기본 특허 통계")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(visuals["yearly_trend"], use_container_width=True)
                    st.plotly_chart(visuals["applicant_trend"], use_container_width=True)
                with col2:
                    st.plotly_chart(visuals["inventor_trend"], use_container_width=True)
                    st.plotly_chart(visuals["status_trend"], use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("총 출원인 수", len(df['applicantName'].unique()))
                with col2:
                    st.metric("총 발명자 수", len(df['inventorName'].unique()))
                with col3:
                    st.metric("등록 특허 비율", f"{(df['registerStatus'] == '등록').mean():.1%}")
                with col4:
                    st.metric("평균 출원-등록 기간", f"{calculate_avg_registration_period(df):.1f}개월")

            with tab2:
                if advanced_analysis:
                    st.subheader("연도별 키워드 트렌드")
                    st.plotly_chart(keyword_trend_fig, use_container_width=True, key='keyword_trend')
                    st.subheader("기술 분야 (IPC) 분석")
                    st.plotly_chart(ipc_fig, use_container_width=True, key='ipc_analysis')
                    st.subheader("기술 성숙도 분석")
                    st.plotly_chart(maturity_fig, use_container_width=True, key='maturity_analysis')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("출원인 협업 네트워크")
                        collaboration_network = analyze_collaboration_network(df, 'applicantName')
                        st.plotly_chart(collaboration_network, use_container_width=True, key='applicant_network')
                    with col2:
                        st.subheader("발명자 협업 네트워크")
                        inventor_network = analyze_collaboration_network(df, 'inventorName')
                        st.plotly_chart(inventor_network, use_container_width=True, key='inventor_network')

            with tab3:
                st.subheader("AI 분석 리포트")
                st.write(ai_analysis)

            with tab4:
                st.subheader("특허 데이터")
                # 데이터 필터링 옵션
                col1, col2, col3 = st.columns(3)
                with col1:
                    status_filter = st.multiselect(
                        "등록상태 필터",
                        options=df['registerStatus'].unique()
                    )
                with col2:
                    year_range = st.slider(
                        "출원연도 범위",
                        min_value=int(df['year'].min()),
                        max_value=int(df['year'].max()),
                        value=(int(df['year'].min()), int(df['year'].max()))
                    )
                with col3:
                    applicant_filter = st.multiselect(
                        "출원인 필터",
                        options=df['applicantName'].unique()
                    )

                # 필터 적용
                filtered_df = filter_patents(df, status_filter, year_range, applicant_filter)
                
                # 데이터프레임 표시
                st.dataframe(
                    filtered_df,
                    column_config={
                        "applicationNumber": "출원번호",
                        "inventionTitle": "발명의 명칭",
                        "applicationDate": "출원일자",
                        "applicantName": "출원인",
                        "inventorName": "발명자",
                        "astrtCont": "요약",
                        "openDate": "공개일자",
                        "openNumber": "공개번호",
                        "publicationDate": "공고일자",
                        "publicationNumber": "공고번호",
                        "registerDate": "등록일자",
                        "registerNumber": "등록번호",
                        "registerStatus": "등록상태"
                    },
                    hide_index=True
                )

                # CSV 다운로드
                csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="CSV 다운로드",
                    data=csv,
                    file_name=f"patent_analysis_{search_query}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()