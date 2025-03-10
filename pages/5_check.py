import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os

# ✅ 배경 이미지 설정 (CSS 활용)
def set_background():
    page_bg_img = """
    <style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1618022325802-7e5e732d97a1?q=80&w=1948&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def run():  # ✅ run() 함수 추가
    # ✅ 배경 이미지 적용
    set_background()

    # Streamlit 대시보드 제목
    st.title("데이터 탐색 대시보드")

    # 📌 사이드바 메뉴
    menu = st.sidebar.radio(
        "탐색할 항목을 선택하세요",
        ["데이터 미리보기", "시계열 시각화"]
    )

    # 📂 1. 로컬 파일에서 데이터 불러오기
    file_path = r"C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\files\MiningProcess2.csv"  # ✅ 로컬 파일 경로 지정
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        st.error("🚨 데이터 파일이 존재하지 않습니다. 파일 경로를 확인하세요.")
        return

    # 🎯 2. 데이터 미리보기
    if menu == "데이터 미리보기":
        st.write("### 데이터 미리보기")
        st.write(df.head())
        st.write("### 데이터 기본 정보")
        st.write(df.describe())

    # 🎯 3. 시계열 데이터 시각화
    elif menu == "시계열 시각화":
        st.write("### 시계열 데이터 시각화")

        # 날짜 컬럼 자동 탐색
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_col = col
                break

        # 예측할 컬럼 선택
        target_col = st.selectbox("예측할 변수 선택", df.columns)

        if not np.issubdtype(df[target_col].dtype, np.number):
            st.error("🚨 선택한 컬럼이 숫자가 아닙니다. 다른 변수를 선택하세요.")
            st.stop()

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df[date_col], df[target_col], label=f"{target_col} 변화")
            ax.set_xlabel("시간")
            ax.set_ylabel(target_col)
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("⚠️ 시계열 데이터를 그릴 'date' 또는 'time' 컬럼이 없습니다.")

# ✅ main.py에서 실행할 수 있도록 run() 함수 추가
if __name__ == "__main__":
    run()
