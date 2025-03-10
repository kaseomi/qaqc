import streamlit as st
import importlib

# ✅ Streamlit 설정 (기본 홈 화면 설정 + 기본 메뉴 비활성화)
st.set_page_config(
    page_title="광산물 대시보드", 
    page_icon="🏠", 
    layout="wide",
    menu_items={"About": None}  # ✅ 기본 메뉴 비활성화
)

# ✅ 사이드바 상단의 자동 생성된 메뉴 숨기기 (CSS 활용)
hide_menu_style = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# ✅ 사이드바에서 표시할 페이지명 (실제 파일명과 매칭)
pages = {
    "대시보드 표지": None, 
    "광산물 예측 모델": "pages.0_model", 
    "광산물 수입 데이터(전체)": "pages.1_income",  
    "광종별 통계 분석(전체)": "pages.2_mineral",  
    "매출 규모별 광산 수(전체)": "pages.3_sales",
    "지역별 광산물 생산량(우리나라)": "pages.4_prod",
}

# ✅ 사이드바에서 보기 좋은 이름으로 페이지 선택
selected_page = st.sidebar.radio("📂 페이지 이동", list(pages.keys()))

# ✅ 선택한 페이지 실행 (대시보드 표지가 아닐 경우)
if selected_page != "대시보드 표지":
    module_name = pages[selected_page]  # 선택한 모듈 이름 가져오기
    if module_name:
        try:
            module = importlib.import_module(module_name)  # ✅ 모듈 강제 불러오기
            importlib.reload(module)  # ✅ 모듈을 새로고침하여 업데이트 반영
            if hasattr(module, "run"):  # ✅ `run()` 함수가 있는 경우 실행
                module.run()
        except Exception as e:
            st.error(f"🚨 오류 발생: {e}")
else:
    # ✅ 배경 이미지 설정 (기존과 동일하게 유지)
    page_bg_img = """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1618022325802-7e5e732d97a1?q=80&w=1948&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        height: 100vh;
        margin: 0;
        padding: 0;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # ✅ 표지 타이틀
    st.markdown("<h1 style='text-align: center; color: white;'>광산물 대시보드</h1>", unsafe_allow_html=True)

    # ✅ 설명 문구 유지
    st.markdown("""
    ### 대시보드 개요
    이 대시보드는 국가별 광산물 여러가지 현황을 분석하고 시각화하는 대시보드입니다.  
    왼쪽 사이드바에서 원하는 페이지를 선택하여 상세 정보를 확인하세요.
    """, unsafe_allow_html=True)

    # ✅ 팀원 정보 유지
    st.markdown("<p style='text-align: left; color: white; font-size: 16px; font-weight: bold;'>4조: 김종범, 김도연, 이기쁨, 강성민</p>", unsafe_allow_html=True)
