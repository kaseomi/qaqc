import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import branca.colormap as cm
import geopandas as gpd
import matplotlib.pyplot as plt
import koreanize_matplotlib

# ✅ 배경 이미지 설정 (Streamlit 전체 페이지 적용)
page_bg_img = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1618022325802-7e5e732d97a1?q=80&w=1948&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def run():
    st.title("대한민국 지역별 광산물 생산량")

    def load_data():
        file_path = r"C:\\Users\\Owner\\Desktop\\VS Code\\심화프로젝트\\4조\\files\\지역별 광산물 생산량.csv"
        df = pd.read_csv(file_path, dtype=str)
        df.drop(columns=['규격'], inplace=True)

        # ✅ '합계' 데이터 제거
        df = df[~df['지역별'].str.contains('합계', na=False)]

        # ✅ 숫자 변환 및 정리
        df['생산량'] = df['2023. 12'].str.replace(',', '', regex=True)
        df['생산량'] = pd.to_numeric(df['생산량'], errors='coerce').fillna(1)  # ✅ 최소 1로 설정하여 색상이 보이도록 조정

        # ✅ CSV 지역명 → JSON 지역명 매핑
        region_name_map = {
            "서울": "서울특별시", "경기": "경기도", "부산": "부산광역시",
            "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시",
            "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시",
            "강원": "강원도", "충북": "충청북도", "충남": "충청남도",
            "전북": "전라북도", "전남": "전라남도", "경북": "경상북도",
            "경남": "경상남도", "제주": "제주특별자치도"
        }

        df['지역별'] = df['지역별'].map(region_name_map)
        df = df.dropna(subset=['지역별'])  # ✅ NaN 값 제거

        # ✅ 지역별 데이터 그룹화
        region_summary = df.groupby('지역별', as_index=False)['생산량'].sum()
        
        # ✅ 생산량 기준으로 정렬 (내림차순 → 왼쪽이 크고, 오른쪽이 작음)
        region_summary = region_summary.sort_values(by='생산량', ascending=False)

        return region_summary

    def plot_map(region_summary):
        # ✅ Folium 지도 생성
        m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles="cartodbpositron")

        # ✅ 로컬 GeoJSON 파일 사용
        geojson_path = r"C:\\Users\\Owner\\Desktop\\VS Code\\심화프로젝트\\4조\\files\\skorea-provinces-2018-geo.json"
        gdf = gpd.read_file(geojson_path)
        gdf = gdf.rename(columns={'name': '지역별'})  # ✅ JSON 컬럼명 일치

        # ✅ 색상 맵핑 설정
        min_value = max(1, region_summary['생산량'].min())
        max_value = region_summary['생산량'].max()
        colormap = cm.LinearColormap(colors=['#EADDC5', '#C2A98B', '#8B5A2B'], vmin=min_value, vmax=max_value)

        for _, row in region_summary.iterrows():
            region = row["지역별"]
            value = row['생산량']
            fill_color = colormap(value)

            if region in gdf["지역별"].values:
                folium.GeoJson(
                    gdf[gdf['지역별'] == region],
                    style_function=lambda x, color=fill_color: {
                        'fillColor': color,
                        'color': 'transparent',
                        'weight': 0,
                        'fillOpacity': 0.6
                    }
                ).add_to(m)

        colormap.caption = "생산량 규모"
        m.add_child(colormap)

        return m

    # ✅ 데이터 불러오기
    data = load_data()
    
    # ✅ 막대그래프와 지도를 좌우 배치로 정렬
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("전국 광산물 생산량 변화")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(data["지역별"].astype(str), data["생산량"].astype(float), color="#8B5A2B")
        ax.set_xlabel("지역")
        ax.set_ylabel("생산량")
        ax.set_title("지역별 총 생산량")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("대한민국 광산물 생산량 지도")
        folium_static(plot_map(data))

if __name__ == "__main__":
    run()
