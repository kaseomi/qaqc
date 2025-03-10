import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def run():  # ✅ run() 함수 추가
    # ✅ 배경 이미지 적용 함수
    def set_background(image_url):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            h1 {{
                text-align: left;
                color: white;
                margin-left: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # ✅ 배경 이미지 설정
    image_url = "https://images.unsplash.com/photo-1618022325802-7e5e732d97a1?q=80&w=1948&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    set_background(image_url)

    # ✅ 제목 표시 (왼쪽 정렬)
    st.markdown("<h1>광종별 수입 비율</h1>", unsafe_allow_html=True)

    # ✅ 데이터 로드
    file_path = r"C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\files\국가별 광종별 수입 위도경도.csv"
    df = pd.read_csv(file_path)

    # ✅ "수입금액" 항목만 필터링하고 "합계" 데이터 제거
    df_filtered = df[(df["수입"] == "수입금액") & (~df["광종"].str.contains("합계"))].copy()
    df_filtered = df_filtered.dropna(subset=["2023 년"])

    # ✅ 광종별 수입금액 합산
    mineral_imports = df_filtered.groupby("광종")["2023 년"].sum().reset_index()

    # ✅ 전체 합 대비 각 광종의 비율 계산
    total_import = mineral_imports["2023 년"].sum()
    mineral_imports["비율(%)"] = (mineral_imports["2023 년"] / total_import) * 100

    # ✅ 상위 광종 선택 (상위 5개)
    mineral_imports = mineral_imports.sort_values(by="비율(%)", ascending=False).head(5)

    # ✅ 삼각형 그래프 데이터 변환
    minerals = mineral_imports["광종"].tolist()
    percentages = mineral_imports["비율(%)"].tolist()
    colors = ['#FFD700', '#4F4F4F', '#FFD700', '#4F4F4F', '#FFD700']  # ✅ 골드 & 다크 그레이 조합

    fig = go.Figure()

    x_positions = [i * 1.2 for i in range(len(minerals))]  # ✅ 삼각형 간격을 줄여서 겹치게 조정

    for i, (mineral, percentage, color, x_pos) in enumerate(zip(minerals, percentages, colors, x_positions)):
        x = [x_pos - 0.6, x_pos, x_pos + 0.6, x_pos - 0.6]  # ✅ 삼각형 X좌표 (겹침 조정)
        y = [0, percentage * 2, 0, 0]  # ✅ 삼각형 Y좌표 (비율을 반영)

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            fill="toself",
            mode="lines",
            line=dict(color=color, width=2),
            hoverinfo="text",
            text=f"{mineral}: {percentage:.1f}%",  # ✅ 그래프에 광종 이름과 비율 표시
            showlegend=False  # ✅ 범례 제거
        ))

        # ✅ 텍스트 라벨 추가 (삼각형 위에 직접 표시)
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[percentage * 2 + 2],  # ✅ Y축 상단에 표시
            text=[f"{mineral}: {percentage:.1f}%"],  # ✅ 라벨을 그래프 내부에 추가
            mode="text",
            textfont=dict(size=16, color="white"),
            showlegend=False  # ✅ 라벨의 범례 제거
        ))

    # ✅ X축 제거 (눈금 & 라벨 숨김)
    fig.update_layout(
        xaxis=dict(
            showticklabels=False,  # ✅ X축 눈금 라벨 숨김
            showgrid=False,  # ✅ X축 그리드 제거
            zeroline=False  # ✅ X축 기준선 제거
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',  
        paper_bgcolor='rgba(0,0,0,0)',
        title="",
        showlegend=False,  # ✅ 전체 범례 제거
        height=600
    )

    # ✅ 그래프 표시
    st.plotly_chart(fig, use_container_width=True)
