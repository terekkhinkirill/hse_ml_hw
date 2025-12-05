import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(
        layout='wide'
)

@st.cache_resource
def get_df():
        return pd.read_csv(Path(__file__).resolve().parent / './X_train_cat.csv')

@st.cache_resource
def get_pair_plot(df):
        return sns.pairplot(df)

@st.cache_resource
def get_corr_heatmap(df):
        fig, _ = plt.subplots(figsize=(4, 4))
        sns.heatmap(df.select_dtypes('number').corr())
        return fig

with open(Path(__file__).resolve().parent / './model_pipeline.pkl', 'rb') as f:
        loaded_pipeline = pickle.load(f)

df = get_df()

with st.expander('Визуализация'):
        if st.checkbox('Визуализировать попарные распределения'):
                st.pyplot(get_pair_plot(df))
        if st.checkbox('Визуализировать попарные корреляции'):
                st.pyplot(get_corr_heatmap(df), use_container_width=False)


with st.expander('Инференс модели'):
        with st.form("Форма предсказания"):
                col_left, col_right = st.columns(2)
                input_data = {}

                with col_left:
                        st.write("**Категориальные:**")
                        for col in df.columns:
                                if df[col].dtype in ('object', 'bool') or col == 'seats':
                                        unique_vals = sorted(df[col].astype(str).unique().tolist())
                                        input_data[col] = st.selectbox(col, unique_vals, key=f"cat_{col}")

                with col_right:
                        st.write("**Числовые:**")
                        for col in df.columns:
                                if df[col].dtype not in ('object', 'bool') and col != 'seats':
                                        if col in ('year', 'km_driven', 'engine'):
                                                val = int(df[col].median())
                                        else:
                                                val = float(df[col].median())
                                        input_data[col] = st.number_input(col, value=val, key=f"num_{col}")

                submitted = st.form_submit_button("Предсказать", use_container_width=True)

        if submitted:
                input_df = pd.DataFrame([input_data])
                st.write(f'Предсказание модели: {loaded_pipeline.predict(input_df[df.columns])[0]}')

with st.expander('Визуализация весов модели'):
        feature_importance_df = pd.DataFrame(
                {
                        'Фича': loaded_pipeline['preprocessor'].get_feature_names_out(),
                        'Коэффициент': loaded_pipeline['ridge'].coef_
                }
        ).sort_values(by='Коэффициент', ascending=False)
        fi = px.bar(feature_importance_df, y='Фича', x='Коэффициент', height=1200)
        st.plotly_chart(fi, use_container_width=True)
