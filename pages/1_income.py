import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px  

def run():  
    def set_background(image_url):
        page_bg = f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        h1, h2, h3 {{
            color: white;
        }}
        .metric-label {{
            font-size: 20px;
            font-weight: bold;
            color: white;
        }}
        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)

    image_url = "https://images.unsplash.com/photo-1618022325802-7e5e732d97a1?q=80&w=1948&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    set_background(image_url)

    file_path = "C:\\Users\\Owner\\Desktop\\VS Code\\심화프로젝트\\4조\\files\\국가별 광종별 수입 위도경도.csv"
    df = pd.read_csv(file_path)

    exchange_rate = 1300
    df["2023 년 원화"] = df["2023 년"] * 1000 * exchange_rate  

    col1, col2 = st.columns([2, 1])  
    with col1:
        st.title("국가별 광산물 수입")
        st.markdown("**아래에서 광종을 선택하면 해당 광종의 수입 정보를 확인할 수 있습니다.**")

    minerals = list(df["광종"].unique())
    selected_mineral = st.selectbox("분석할 광종을 선택하세요:", minerals)

    df_selected_mineral = df[(df["광종"] == selected_mineral) & (df["수입"] == "수입금액")]
    total_import_amount = df_selected_mineral["2023 년 원화"].sum()  
    
    조 = total_import_amount // 1_0000_0000_0000
    억 = (total_import_amount % 1_0000_0000_0000) // 1_0000_0000

    if total_import_amount >= 1_0000_0000_0000:
        formatted_amount = f"{조}조 {억}억 원"
    elif total_import_amount >= 1_0000_0000:
        formatted_amount = f"{억}억 원"
    else:
        formatted_amount = f"{total_import_amount:,.0f} 원"

    with col2:
        st.metric(label=f"2023년 {selected_mineral} 총 수입금액", value=formatted_amount)

    left_col, right_col = st.columns([1.5, 1])
    with left_col:
        st.subheader(f"{selected_mineral} 수입 비율 지도")
        df_filtered = df[(df["광종"] == selected_mineral) & (df["수입"] == "수입%")]
        df_filtered = df_filtered[df_filtered["2023 년"] > 0]  

        max_radius = 600000
        min_radius = 10000
        df_filtered["원크기"] = ((df_filtered["2023 년"] / df_filtered["2023 년"].max()) * max_radius) + min_radius  

        if not df_filtered.empty:
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_filtered,
                get_position=["경도", "위도"],
                get_radius="원크기",
                get_fill_color="[255, 215, 0, 180]",
                pickable=True,
                auto_highlight=True,
            )

            map_deck = pdk.Deck(
                layers=[scatter_layer],
                initial_view_state=pdk.ViewState(
                    latitude=20,
                    longitude=0,
                    zoom=1.5,
                    pitch=0,
                    bearing=0,
                ),
                map_style="mapbox://styles/mapbox/light-v10"
            )
            st.pydeck_chart(map_deck, use_container_width=False)
        else:
            st.warning("⚠ 선택한 광종의 수입 데이터가 없습니다.")

    with right_col:
        st.subheader("상위 5개국 수입 비율(%)")
        df_filtered["2023 년"] = df_filtered.groupby("국가")["2023 년"].transform("sum")
        top_5_countries = df_filtered.nlargest(5, "2023 년")
        total_percentage = df_filtered["2023 년"].sum()
        top_5_countries["2023 년"] = (top_5_countries["2023 년"] / total_percentage) * 100

        if not top_5_countries.empty:
            fig = px.bar(
                top_5_countries,
                x="국가",
                y="2023 년",
                text="2023 년",
                color="국가",
                color_discrete_sequence=["#FFD700", "#4F4F4F"]
            )
            fig.update_traces(textposition="outside", texttemplate="%{text:.2f}%")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="",
                yaxis_title="비율 (%)",
                title="",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠ 상위 5개국 데이터를 찾을 수 없습니다.")
