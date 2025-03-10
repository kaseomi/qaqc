import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import torch
import torch.nn as nn
import plotly.express as px
from torch.utils.data import DataLoader, TensorDataset

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2): 
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.bn(lstm_out[:, -1, :])
        return self.linear(x)

# 사이드바에 페이지 선택 옵션 추가
page = st.sidebar.selectbox("페이지 선택", ["데이터 변환", "예측 모델"])

if page == "데이터 변환":
    st.title("Flotation Column Ratio 변환 및 예측")
    file_path = r"C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\files\MiningProcess.csv"  # 여기에 로컬 파일 경로를 입력하세요
    data = pd.read_csv(file_path)
    st.write("원본 데이터:", data.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Ratio 변환 시작"):
            transformed_data = data.copy()
            
            # Air Flow 합계 계산
            air_flow_cols = [f'Flotation Column {str(i).zfill(2)} Air Flow' for i in range(1, 8)]
            air_flow_sum = transformed_data[air_flow_cols].sum(axis=1)
            
            # Level 합계 계산
            level_cols = [f'Flotation Column {str(i).zfill(2)} Level' for i in range(1, 8)]
            level_sum = transformed_data[level_cols].sum(axis=1)
            
            # Ratio 계산 및 새로운 컬럼 추가
            for col in air_flow_cols:
                ratio_col = f'{col} Ratio'
                transformed_data[ratio_col] = transformed_data[col] / air_flow_sum
            
            for col in level_cols:
                ratio_col = f'{col} Ratio'
                transformed_data[ratio_col] = transformed_data[col] / level_sum
            
            # 기존 컬럼 삭제
            transformed_data = transformed_data.drop(air_flow_cols + level_cols, axis=1)
            
            st.session_state['transformed_data'] = transformed_data
            st.write("변환된 데이터:", transformed_data.head())

    with col2:
        if 'transformed_data' in st.session_state:
            model_choice = st.selectbox(
                "예측 모델 선택",
                ["Decision Tree", "XGBoost", "LSTM"]
            )
            
            if st.button("예측 시작"):
                data = st.session_state['transformed_data']
                X = data.drop(['date', '% Iron Concentrate', '% Silica Concentrate'], axis=1)
                y = data['% Silica Concentrate']
                # 데이터 스케일링
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 모델 예측
                if model_choice == "Decision Tree":
                    with open(r'C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\pages\dt_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    st.write("Decision Tree 모델 로드 완료")
                    
                    predictions = model.predict(X_scaled)
                
                elif model_choice == "XGBoost":
                    with open(r'C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\pages\xgb_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    st.write("XGBoost 모델 로드 완료")
                    
                    predictions = model.predict(X_scaled)
                
                elif model_choice == "LSTM":
                    # 디바이스 설정
                    device = torch.device("cpu")

                    # 모델 로드
                    def load_model():
                        model = LSTMRegressor(input_size=21).to(device)  # input_size는 실제 데이터 컬럼 수에 맞춰 조정
                        model.load_state_dict(torch.load(r"C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\pages\lstm_model.pth", map_location=device))
                        model.eval()
                        return model
                    
                    # 시퀀스 데이터 생성 함수
                    def create_sequences(X, time_steps=3):
                        Xs = []
                        for i in range(len(X) - time_steps):
                            Xs.append(X[i:(i + time_steps)])
                        return np.array(Xs)
                    
                    try:
                        X = data.drop(["date", "% Iron Concentrate", "% Silica Concentrate"], axis=1)
                        y = data["% Silica Concentrate"]
                        
                        scaler_X = StandardScaler()
                        scaler_y = StandardScaler()
                        X_scaled = scaler_X.fit_transform(X)
                        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
                        
                        time_steps = 3
                        X_seq = create_sequences(X_scaled, time_steps)
                        
                        X_seq_tensor = torch.FloatTensor(X_seq)
                        dataset = TensorDataset(X_seq_tensor)
                        batch_size = 64
                        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                        
                        model = load_model()
                        y_pred_scaled = []

                        with torch.no_grad():
                            for batch in data_loader:
                                batch_X = batch[0].to(device)
                                batch_pred = model(batch_X).cpu().numpy()
                                y_pred_scaled.append(batch_pred)

                        y_pred_scaled = np.vstack(y_pred_scaled)
                        y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
                        y_actual_original = y[time_steps:].values
                        
                        results = pd.DataFrame({
                            "Date": data["date"].iloc[time_steps:].values,
                            "Actual": y_actual_original.flatten(),
                            "Predicted": y_pred_original.flatten()
                        })
                        
                        fig = px.line(results, x='Date', y=['Actual', 'Predicted'],
                                    title='실제값 vs 예측값')
                        st.plotly_chart(fig)
                        
                        # 성능 지표 계산
                        mse = np.mean((results['Actual'] - results['Predicted'])**2)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(results['Actual'], results['Predicted'])
                        
                        st.write(f"MSE: {mse:.4f}")
                        st.write(f"RMSE: {rmse:.4f}")
                        st.write(f"R² Score: {r2:.4f}")
                    
                    except Exception as e:
                        st.error(f"오류 발생: {e}")

                # 결과 시각화 및 성능 지표 계산 (Decision Tree 및 XGBoost)
                if model_choice in ["Decision Tree", "XGBoost"]:
                    results = pd.DataFrame({
                        "Date": data["date"],
                        "Actual": y,
                        "Predicted": predictions
                    })
                    
                    fig = px.line(results, x='Date', y=['Actual', 'Predicted'],
                                title='실제값 vs 예측값')
                    st.plotly_chart(fig)
                    
                    # 성능 지표 계산
                    mse = np.mean((results['Actual'] - results['Predicted'])**2)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(results['Actual'], results['Predicted'])
                    
                    st.write(f"MSE: {mse:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")
                    st.write(f"R² Score: {r2:.4f}")

# 예측 모델 페이지
if page == "예측 모델":
    st.title("철광석 품질 예측 모델")

    if "transformed_data" not in st.session_state:
        st.warning("먼저 '데이터 변환' 페이지에서 데이터를 변환하세요.")
    else:
        data = st.session_state['transformed_data']
        X = data.drop(['date', '% Iron Concentrate', '% Silica Concentrate'], axis=1)
        y = data['% Silica Concentrate']

        # 데이터 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_choice = st.selectbox("예측 모델 선택", ["Decision Tree", "XGBoost"])

        if st.button("예측 시작"):
            try:
                if model_choice == "Decision Tree":
                    with open(r'C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\pages\dt_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    predictions = model.predict(X_scaled)

                elif model_choice == "XGBoost":
                    with open(r'C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\pages\xgb_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    predictions = model.predict(X_scaled)

            except Exception as e:
                st.error(f"예측 중 오류 발생: {str(e)}")

        st.write("새로운 데이터를 입력하세요:")
        new_data = {}

        # 입력 순서 지정
        input_order = [
            '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 
            'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density'
        ] 

        # 슬라이더 설정
        def create_slider(col, min_val, max_val):
            return st.slider(f"{col} (최소: {min_val}, 최대: {max_val})", min_value=float(min_val), max_value=float(max_val), value=float(min_val), key=col)

        # 일반 입력 값 설정
        for col in input_order:
            if col == 'Ore Pulp Flow':
                new_data[col] = st.select_slider(f"{col} (선택 가능 값: 380, 400)", options=[380, 400], value=380, key=col)
            else:
                min_val = data[col].min()
                max_val = data[col].max()
                new_data[col] = create_slider(col, min_val, max_val)

        # Air Flow 설정
        air_flow_cols = [f'Flotation Column {str(i).zfill(2)} Air Flow' for i in range(1, 8)]
        air_flow_values = {}

        air_flow_options = {
            'Flotation Column 01 Air Flow': [250, 300],
            'Flotation Column 02 Air Flow': [250, 300],
            'Flotation Column 03 Air Flow': [250, 300],
            'Flotation Column 04 Air Flow': [250, 300],
            'Flotation Column 05 Air Flow': [250, 300],
            'Flotation Column 06 Air Flow': [250, 300, 350],
            'Flotation Column 07 Air Flow': [250, 300]
        }

        for col in air_flow_cols:
            air_flow_values[col] = st.select_slider(f"{col} (선택 가능 값: {air_flow_options[col]})", options=sorted(air_flow_options[col]), value=air_flow_options[col][0], key=col)

        # Air Flow 합산
        air_flow_sum = sum(air_flow_values.values())

        # Level 설정
        level_cols = [f'Flotation Column {str(i).zfill(2)} Level' for i in range(1, 8)]
        level_values = {}

        level_options = {
            'Flotation Column 01 Level': [400, 450, 500, 600, 700, 800],
            'Flotation Column 02 Level': [400, 450, 500, 550, 600, 700, 800],
            'Flotation Column 03 Level': [400, 450, 500, 550, 600, 700, 800],
            'Flotation Column 04 Level': [350, 400, 450, 500],
            'Flotation Column 05 Level': [350, 400, 450, 500],
            'Flotation Column 06 Level': [350, 400, 450, 500, 600],
            'Flotation Column 07 Level': [350, 400, 450, 500]
        }

        for col in level_cols:
            level_values[col] = st.select_slider(f"{col} (선택 가능 값: {level_options[col]})", options=sorted(level_options[col]), value=level_options[col][0], key=col)

        # 데이터 입력 순서 조정
        for col in air_flow_cols:
            new_data[f"{col} Ratio"] = air_flow_values[col] / air_flow_sum

        level_sum = sum(level_values.values())
        for col in level_cols:
            new_data[f"{col} Ratio"] = level_values[col] / level_sum

        # 예측 버튼
        if st.button("새로운 데이터 예측하기"):
            try:
                new_data_df = pd.DataFrame([new_data])
                new_data_scaled = scaler.transform(new_data_df)
                
                if model_choice == "Decision Tree":
                    with open(r'C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\pages\dt_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    prediction = model.predict(new_data_scaled)
                elif model_choice == "XGBoost":
                    with open(r'C:\Users\Owner\Desktop\VS Code\심화프로젝트\4조\pages\xgb_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    prediction = model.predict(new_data_scaled)
                
                st.write(f"예측된 % Silica Concentrate: {prediction[0]:.4f}")
            except Exception as e:
                st.error(f"예측 중 오류 발생: {str(e)}")