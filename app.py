import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Устанавливаем конфигурацию страницы
st.set_page_config(page_title="Система прогнозирования продаж", layout="wide", initial_sidebar_state="expanded")

# --- Стили и фон ---
def set_background():
    """ Устанавливает фон и пользовательские стили для приложения. """
    page_bg_img = '''
    <style>
    .stApp {
        background-image: linear-gradient(to right top, #d16ba5, #c777b9, #ba83ca, #aa8fd8, #9a9ae1, #8aa7ec, #79b3f4, #69bff8, #52cffe, #41dfff, #46eefa, #5ffbf1);
        background-size: cover;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 2rem;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background()

# --- Функции для работы с данными ---
@st.cache_data
def load_and_validate_data(uploaded_file):
    """ Загружает и проверяет данные из Excel файла. """
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Отсутствуют колонки: {missing_cols}")
            return None
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Datasales']).sort_values('Datasales')
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None

def show_data_statistics(df):
    """ Отображает основную статистику по данным. """
    st.subheader("📊 Статистика данных")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего записей", f"{len(df):,}")
    with col2:
        st.metric("Уникальных товаров", f"{df['Art'].nunique():,}")
    with col3:
        st.metric("Магазинов", f"{df['Magazin'].nunique():,}")
    with col4:
        st.metric("Сегментов", f"{df['Segment'].nunique():,}")
    st.write(f"**Период данных:** с {df['Datasales'].min().date()} по {df['Datasales'].max().date()}")

def create_features(df):
    """ Создает новые признаки для модели CatBoost. """
    df = df.copy().sort_values(['Magazin', 'Segment', 'Art', 'Datasales']).reset_index(drop=True)
    df['year'] = df['Datasales'].dt.year
    df['month'] = df['Datasales'].dt.month
    df['dayofweek'] = df['Datasales'].dt.dayofweek
    df['quarter'] = df['Datasales'].dt.quarter
    df['is_weekend'] = (df['Datasales'].dt.dayofweek >= 5).astype(int)
    for col in ['Qty', 'Sum']:
        for lag in [1, 7, 30]:
            df[f'{col}_lag_{lag}'] = df.groupby(['Magazin', 'Segment', 'Art'])[col].shift(lag)
        for window in [7, 30]:
            df[f'{col}_ma_{window}'] = df.groupby(['Magazin', 'Segment', 'Art'])[col].transform(lambda x: x.rolling(window).mean())
    return df

# --- Функции для моделей ---
def prepare_prophet_data(df, target_col='Qty'):
    """ Подготавливает данные для модели Prophet. """
    prophet_df = df.groupby('Datasales')[target_col].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

@st.cache_resource
def train_prophet_model(train_data, periods=30):
    """ Обучает модель Prophet и делает предсказание. """
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def prepare_catboost_data(df):
    """ Подготавливает данные для модели CatBoost. """
    df_features = create_features(df)
    daily_data = df_features.groupby('Datasales').agg({
        'Qty': 'sum', 'Sum': 'sum', 'Price': 'mean',
        'year': 'first', 'month': 'first', 'dayofweek': 'first',
        'quarter': 'first', 'is_weekend': 'first'
    }).reset_index()
    for lag in [1, 7, 30]:
        daily_data[f'Qty_lag_{lag}'] = daily_data['Qty'].shift(lag)
    for window in [7, 30]:
        daily_data[f'Qty_ma_{window}'] = daily_data['Qty'].rolling(window).mean()
    return daily_data

@st.cache_resource
def train_catboost_model(data, periods=30):
    """ Обучает модель CatBoost и делает предсказание. """
    feature_cols = ['year', 'month', 'dayofweek', 'quarter', 'is_weekend', 'Price',
                   'Qty_lag_1', 'Qty_lag_7', 'Qty_lag_30', 'Qty_ma_7', 'Qty_ma_30']
    clean_data = data.dropna()
    if len(clean_data) < 30:
        raise ValueError("Недостаточно данных для обучения CatBoost")
    X, y = clean_data[feature_cols], clean_data['Qty']
    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6,
                             loss_function='RMSE', random_seed=42, verbose=False)
    model.fit(X, y)
    last_date = data['Datasales'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    forecast_data = []
    last_row = clean_data.iloc[-1].copy()
    for i, date in enumerate(future_dates):
        last_row.update({
            'year': date.year, 'month': date.month, 'dayofweek': date.dayofweek,
            'quarter': date.quarter, 'is_weekend': 1 if date.dayofweek >= 5 else 0
        })
        pred = max(0, model.predict([last_row[feature_cols]])[0])
        forecast_data.append({'ds': date, 'yhat': pred})
        if i < periods - 1:
            last_row['Qty_lag_30'] = last_row['Qty_lag_7'] if i >= 23 else last_row['Qty_lag_30']
            last_row['Qty_lag_7'] = last_row['Qty_lag_1'] if i >= 6 else last_row['Qty_lag_7']
            last_row['Qty_lag_1'] = pred
            if i >= 6:
                last_row['Qty_ma_7'] = (last_row['Qty_ma_7'] * 6 + pred) / 7
            if i >= 29:
                last_row['Qty_ma_30'] = (last_row['Qty_ma_30'] * 29 + pred) / 30
    return model, pd.DataFrame(forecast_data)

# --- Функции для визуализации ---
def plot_forecast(df, forecast, model_type, title):
    """ Строит график фактических данных и прогноза. """
    fig = go.Figure()

    # Фактические данные
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'], mode='lines+markers', name='Факт',
        line=dict(color='#1f77b4', width=2), marker=dict(size=5)
    ))

    # Прогноз
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))

    # Доверительный интервал для Prophet
    if model_type == 'Prophet' and 'yhat_lower' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines',
            line_color='rgba(255,127,14,0.3)', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines',
            line_color='rgba(255,127,14,0.3)', name='Доверительный интервал'
        ))

    fig.update_layout(
        title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title='Дата', yaxis_title='Количество',
        hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_kpi(df):
    """ Отображает KPI-метрики. """
    st.subheader("🚀 Ключевые показатели")
    total_sales = df['Sum'].sum()
    avg_receipt = df['Sum'].mean()
    unique_products = df['Art'].nunique()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Общий объем продаж (грн)", f"{total_sales:,.0f}")
    with col2:
        st.metric("Средний чек (грн)", f"{avg_receipt:,.2f}")
    with col3:
        st.metric("Количество уникальных товаров", f"{unique_products:,}")

def plot_segment_distribution(df):
    """ Строит круговую диаграмму распределения продаж по сегментам. """
    st.subheader("🍩 Распределение продаж по сегментам")
    segment_sales = df.groupby('Segment')['Sum'].sum().reset_index()
    fig = px.pie(segment_sales, values='Sum', names='Segment', title='Доля каждого сегмента в общей выручке',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def plot_weekday_sales(df):
    """ Строит гистограмму распределения продаж по дням недели. """
    st.subheader("📅 Продажи по дням недели")
    df['weekday'] = df['Datasales'].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_sales = df.groupby('weekday')['Qty'].sum().reindex(weekday_order).reset_index()
    fig = px.bar(weekday_sales, x='weekday', y='Qty', title='Общее количество проданных товаров по дням недели',
                 labels={'weekday': 'День недели', 'Qty': 'Количество'})
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)


# --- Основное приложение ---
st.title("🏪 Система прогнозирования продаж")
uploaded_file = st.file_uploader("Загрузите Excel файл", type=['xlsx', 'xls'])

if uploaded_file:
    df = load_and_validate_data(uploaded_file)
    if df is not None:
        show_data_statistics(df)
        plot_kpi(df)

        col1, col2 = st.columns(2)
        with col1:
            plot_segment_distribution(df)
        with col2:
            plot_weekday_sales(df)

        st.sidebar.header("Параметры прогноза")
        selected_magazin = st.sidebar.selectbox("Выберите магазин", ['Все'] + list(df['Magazin'].unique()))
        selected_segment = st.sidebar.selectbox("Выберите сегмент", ['Все'] + list(df['Segment'].unique()))
        model_type = st.sidebar.selectbox("Модель прогнозирования", ['Prophet', 'CatBoost'])
        forecast_days = st.sidebar.slider("Период прогноза (дней)", 7, 90, 30)

        if st.sidebar.button("🔮 Показать прогноз", type="primary"):
            filtered_df = df.copy()
            if selected_magazin != 'Все':
                filtered_df = filtered_df[filtered_df['Magazin'] == selected_magazin]
            if selected_segment != 'Все':
                filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]

            if len(filtered_df) < 10:
                st.error("Недостаточно данных для прогнозирования (минимум 10 записей)")
            else:
                with st.spinner(f'Обучение модели {model_type}...'):
                    try:
                        if model_type == 'Prophet':
                            prophet_data = prepare_prophet_data(filtered_df)
                            model, forecast = train_prophet_model(prophet_data, periods=forecast_days)
                        else:
                            catboost_data = prepare_catboost_data(filtered_df)
                            model, forecast = train_catboost_model(catboost_data, periods=forecast_days)
                            prophet_data = prepare_prophet_data(filtered_df)
                    except Exception as e:
                        st.error(f"Ошибка обучения модели: {e}")
                        st.stop()

                st.subheader(f"📊 Прогноз продаж на {forecast_days} дней")
                fig = plot_forecast(prophet_data, forecast, model_type, f"Прогноз продаж - {model_type}")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Загрузите файл Excel с данными о продажах для начала работы")
    st.subheader("📋 Требования к формату данных")
    st.markdown("""
    **Обязательные колонки:**
    - `Magazin` - название магазина
    - `Datasales` - дата продажи
    - `Art` - артикул товара
    - `Describe` - описание товара
    - `Model` - модель товара
    - `Segment` - сегмент товара
    - `Price` - цена
    - `Qty` - количество
    - `Sum` - сумма продажи

    **Доступные модели:**
    - **Prophet** - статистическая модель для анализа временных рядов.
    - **CatBoost** - модель машинного обучения для более точного прогнозирования.
    """)
