import streamlit as st
import pandas as pd
import plotly.graph_objects as go  # ✅ Plotly 추가

def run():  # ✅ `main.py`에서 import하여 실행할 수 있도록 함수화
    # ✅ 배경 이미지 적용 함수
    def set_background(image_url):
        page_bg = f"""
        <style>
        .stApp {{
            background-image: url("{image_url}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
        }}
        h1 {{
            font-size: 36px;
            font-weight: bold;
            color: white !important;
            margin-left: 20px;
        }}
        .stPlotlyChart {{
            background: transparent !important;
        }}
        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)

    # ✅ 새로운 배경 이미지 적용 (사용자가 요청한 이미지)
    image_url = "https://images.unsplash.com/photo-1618022325802-7e5e732d97a1?q=80&w=1948&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    set_background(image_url)

    # ✅ 타이틀을 왼쪽 위에 크게 표시
    st.markdown("<h1> 매출 규모별 광산 수</h1>", unsafe_allow_html=True)

    # ✅ 데이터 로드
    file_path = r"C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\files\매출규모별광산수.csv"
    df = pd.read_csv(file_path)

    # ✅ "일반광합계" 제거
    df = df[df["광종(1)"] != "일반광합계"]

    # ✅ "매출규모별(1)"에서 "합계" 제거
    df = df[df["매출규모별(1)"] != "합계"]

    # ✅ 컬럼명 변환 (시각화에 적합하게 변경)
    df.rename(columns={
        "광종(1)": "광종",
        "매출규모별(1)": "매출 규모",
        "2021": "광산 수"
    }, inplace=True)

    # ✅ Y축 범주 순서 정렬
    df["매출 규모"] = pd.Categorical(df["매출 규모"], categories=df["매출 규모"].unique(), ordered=True)

    # ✅ 데이터 그룹화 (광종별 매출 규모별 광산 수)
    grouped_df = df.pivot_table(index="매출 규모", columns="광종", values="광산 수", aggfunc="sum", fill_value=0)

    # ✅ 시각화를 위한 색상 지정 (이미지와 유사한 색상)
    colors = {"금속광": "#D2691E", "비금속광": "#FFD700", "석탄광": "#BEBEBE"}  

    # ✅ 원하는 순서대로 막대 그래프 추가 (금속광 → 비금속광 → 석탄광 순서 강제)
    fig = go.Figure()

    for 광종 in ["금속광", "비금속광", "석탄광"]:  # ✅ 강제 순서 지정
        values = grouped_df[광종]

        # ✅ 값이 0이면 라벨을 숨김
        text_labels = [str(v) if v > 0 else "" for v in values]

        fig.add_trace(go.Bar(
            y=grouped_df.index,  # ✅ 매출 규모를 Y축으로 설정
            x=values,  
            name=광종,
            orientation="h",  # ✅ 가로 막대 그래프 설정
            marker=dict(color=colors[광종]),  
            text=text_labels,  
            textposition="outside",  # ✅ 개별 광산 수를 막대 끝에 표시
            textfont_size=14,  
            texttemplate="%{text}",
        ))

    # ✅ 그래프 레이아웃 설정 (배경을 투명하게 유지 & 범례 크기 조정)
    fig.update_layout(
        xaxis_title="광산 수 (단위: 개소)",
        yaxis_title="매출 규모",
        xaxis=dict(tickfont=dict(size=14, color="white")),
        yaxis=dict(tickfont=dict(size=14, color="white")),
        barmode="group",  # ✅ 그룹 형태로 배치
        legend_title="광종",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',  # ✅ 그래프 배경 투명하게 설정
        paper_bgcolor='rgba(0,0,0,0)',  # ✅ 전체 배경 투명하게 설정
        legend=dict(
            font=dict(size=16, color="white"),  # ✅ 범례 크기 조정 (기존보다 큼)
            orientation="h",  # ✅ 범례를 가로로 정렬
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        )
    )

    # ✅ Streamlit에 그래프 표시
    st.plotly_chart(fig, use_container_width=True)
