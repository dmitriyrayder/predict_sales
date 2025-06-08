import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Система предсказания продаж", layout="wide")

def load_and_validate_data(uploaded_file):
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
    st.subheader("📊 Статистика данных")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего записей", len(df))
    with col2:
        st.metric("Уникальных товаров", df['Art'].nunique())
    with col3:
        st.metric("Магазинов", df['Magazin'].nunique())
    with col4:
        st.metric("Сегментов", df['Segment'].nunique())
    st.write("**Период данных:**", f"{df['Datasales'].min().date()} - {df['Datasales'].max().date()}")

def create_features(df):
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

def prepare_prophet_data(df, target_col='Qty'):
    prophet_df = df.groupby('Datasales')[target_col].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_prophet_model(train_data, periods=30):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def prepare_catboost_data(df):
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

def train_catboost_model(data, periods=30):
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

def calculate_segment_volatility(df, selected_magazin, selected_segment):
    filtered = df.copy()
    if selected_magazin != 'Все':
        filtered = filtered[filtered['Magazin'] == selected_magazin]
    if selected_segment != 'Все':
        filtered = filtered[filtered['Segment'] == selected_segment]
    daily_sales = filtered.groupby('Datasales')['Qty'].sum()
    if len(daily_sales) < 2:
        return 0.2
    volatility = daily_sales.std() / daily_sales.mean() if daily_sales.mean() > 0 else 0.2
    return max(0.1, min(0.5, volatility))

def get_forecast_scenarios(forecast, model_type, segment_volatility=0.2):
    realistic = forecast['yhat'].values
    if model_type == 'Prophet' and 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        optimistic = forecast['yhat_upper'].values
        pessimistic = forecast['yhat_lower'].values
    else:
        pessimistic = realistic * (1 - segment_volatility)
        optimistic = realistic * (1 + segment_volatility * 0.7)
    return realistic, optimistic, np.maximum(pessimistic, 0)

def get_top_models_by_segment(df, selected_magazin):
    if selected_magazin != 'Все':
        df = df[df['Magazin'] == selected_magazin]
    segments_top_models = {}
    for segment in df['Segment'].unique():
        segment_data = df[df['Segment'] == segment]
        model_stats = segment_data.groupby('Model').agg({
            'Qty': 'sum',
            'Sum': 'sum',
            'Price': 'mean'
        }).reset_index()
        top_models = model_stats.nlargest(10, 'Qty')
        top_models['Price'] = top_models['Price'].round(0)
        top_models['Qty'] = top_models['Qty'].astype(int)
        top_models['Sum'] = top_models['Sum'].round(0)
        segments_top_models[segment] = top_models
    return segments_top_models

def show_forecast_statistics(filtered_df, forecast, forecast_days, selected_magazin, selected_segment, model_type, df):
    st.subheader("📊 Статистика прогноза")
    historical_data = filtered_df.groupby('Datasales')['Qty'].sum().reset_index()
    
    # Правильный расчет исторических данных за аналогичный период
    if len(historical_data) >= forecast_days:
        hist_qty = historical_data.tail(forecast_days)['Qty'].sum()
        period_start = historical_data.tail(forecast_days)['Datasales'].min().date()
        period_end = historical_data.tail(forecast_days)['Datasales'].max().date()
    else:
        # Если данных меньше периода прогноза, берем среднее за день и экстраполируем
        daily_avg = historical_data['Qty'].mean() if len(historical_data) > 0 else 0
        hist_qty = daily_avg * forecast_days
        period_start = historical_data['Datasales'].min().date() if len(historical_data) > 0 else "N/A"
        period_end = historical_data['Datasales'].max().date() if len(historical_data) > 0 else "N/A"

    avg_price = filtered_df['Price'].mean()
    hist_sum = hist_qty * avg_price
    segment_volatility = calculate_segment_volatility(df, selected_magazin, selected_segment)
    forecast_period = forecast.tail(forecast_days) if len(forecast) > forecast_days else forecast
    realistic, optimistic, pessimistic = get_forecast_scenarios(forecast_period, model_type, segment_volatility)
    realistic_qty, optimistic_qty, pessimistic_qty = realistic.sum(), optimistic.sum(), pessimistic.sum()
    realistic_sum, optimistic_sum, pessimistic_sum = realistic_qty * avg_price, optimistic_qty * avg_price, pessimistic_qty * avg_price

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Количество (шт.)")
        qty_data = pd.DataFrame({
            'Сценарий': ['Исторический', 'Пессимистичный', 'Реальный', 'Оптимистичный'],
            'Количество': [int(hist_qty), int(pessimistic_qty), int(realistic_qty), int(optimistic_qty)],
            'Изменение': ['-'] + [f"{((qty/hist_qty - 1) * 100):+.1f}%" if hist_qty > 0 else '-' 
                                 for qty in [pessimistic_qty, realistic_qty, optimistic_qty]]
        })
        st.dataframe(qty_data, hide_index=True)
    
    with col2:
        st.markdown("### 💰 Сумма (ГРН.)")
        sum_data = pd.DataFrame({
            'Сценарий': ['Исторический', 'Пессимистичный', 'Реальный', 'Оптимистичный'],
            'Сумма': [f"{hist_sum:,.0f}", f"{pessimistic_sum:,.0f}", f"{realistic_sum:,.0f}", f"{optimistic_sum:,.0f}"],
            'Изменение': ['-'] + [f"{((sum_val/hist_sum - 1) * 100):+.1f}%" if hist_sum > 0 else '-' 
                                 for sum_val in [pessimistic_sum, realistic_sum, optimistic_sum]]
        })
        st.dataframe(sum_data, hide_index=True)
    
    # Комментарий о периоде исторических данных
    st.info(f"📊 **Волатильность сегмента**: {segment_volatility:.1%}")
    st.info(f"📅 **Исторические данные за период**: {period_start} - {period_end} ({forecast_days} дней)")
    st.markdown(f"**Фильтры:** Магазин: {selected_magazin}, Сегмент: {selected_segment}, Период: {forecast_days} дней")

def generate_insights_for_magazin(df, forecast_data, model_type, selected_magazin):
    insights = []
    problems = []
    if selected_magazin != 'Все':
        magazin_df = df[df['Magazin'] == selected_magazin]
    else:
        magazin_df = df
    magazin_prophet_data = prepare_prophet_data(magazin_df)
    if len(magazin_prophet_data) >= 30:
        recent_data = magazin_prophet_data.tail(30)['y'].mean()
        older_data = magazin_prophet_data.iloc[-60:-30]['y'].mean() if len(magazin_prophet_data) >= 60 else recent_data
        if recent_data > older_data * 1.1:
            insights.append("📈 **РОСТ ПРОДАЖ**: Наблюдается рост продаж на 10%+. Рекомендуется увеличить закупки.")
    if len(magazin_prophet_data) > 1:
        volatility = magazin_prophet_data['y'].std() / magazin_prophet_data['y'].mean() if magazin_prophet_data['y'].mean() > 0 else 0
        if volatility > 0.5:
            insights.append("⚠️ **ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ**: Рекомендуется увеличить страховые запасы.")
            problems.append("⚠️ **ПРОБЛЕМА**: Высокая нестабильность продаж затрудняет планирование")
        elif volatility < 0.1:
            insights.append("✅ **СТАБИЛЬНЫЕ ПРОДАЖИ**: Низкая волатильность позволяет точнее планировать.")
    if len(forecast_data) > 14:
        forecast_start = forecast_data.iloc[:7]['yhat'].mean()
        forecast_end = forecast_data.iloc[-7:]['yhat'].mean()
        forecast_trend = forecast_end / forecast_start if forecast_start > 0 else 1
        if forecast_trend > 1.05:
            insights.append("🚀 **ПРОГНОЗ РОСТА**: Ожидается рост продаж. Подготовьтесь к увеличению спроса.")
        elif forecast_trend < 0.95:
            insights.append("⬇️ **ПРОГНОЗ СНИЖЕНИЯ**: Ожидается снижение продаж. Рассмотрите промо-активности.")
            problems.append("📉 **БУДУЩАЯ ПРОБЛЕМА**: Прогнозируется снижение продаж")
    if not insights:
        insights.append("📊 **МОНИТОРИНГ**: Продолжайте отслеживать показетели для выявления трендов.")
    return insights, problems

def plot_forecast(df, forecast, model_type, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Фактические данные', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз', line=dict(color='red')))
    if model_type == 'Prophet' and 'yhat_lower' in forecast.columns:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Доверительный интервал'))
    fig.update_layout(title=title, xaxis_title='Дата', yaxis_title='Количество', hovermode='x unified')
    return fig

def plot_prophet_components(model, forecast):
    """Создает графики компонентов Prophet"""
    from prophet.plot import plot_components_plotly
    fig = plot_components_plotly(model, forecast)
    return fig

def plot_prophet_seasonality(forecast):
    """Создает график сезонности Prophet"""
    fig = go.Figure()
    
    # Недельная сезонность
    if 'weekly' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['weekly'], 
            mode='lines', 
            name='Недельная сезонность',
            line=dict(color='green')
        ))
    
    # Годовая сезонность
    if 'yearly' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['yearly'], 
            mode='lines', 
            name='Годовая сезонность',
            line=dict(color='orange')
        ))
    
    fig.update_layout(
        title='Компонеты сезонности Prophet',
        xaxis_title='Дата',
        yaxis_title='Влияние',
        hovermode='x unified'
    )
    return fig

def plot_monthly_revenue_by_segment(df, selected_magazin, selected_segment):
    filtered_df = df.copy()
    if selected_magazin != 'Все':
        filtered_df = filtered_df[filtered_df['Magazin'] == selected_magazin]
    if selected_segment != 'Все':
        filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]
    
    filtered_df['year_month'] = filtered_df['Datasales'].dt.to_period('M')
    monthly_revenue = filtered_df.groupby('year_month')['Sum'].sum().reset_index()
    monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)
    
    fig = px.bar(monthly_revenue, x='year_month', y='Sum', 
                title=f'Продажи в деньгах по месяцам - {selected_segment}',
                labels={'year_month': 'Месяц', 'Sum': 'Выручка (ГРН.)'},
                text='Sum')
    
    # Форматирование подписей с суммами
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title='Месяц', yaxis_title='Выручка (ГРН.)', hovermode='x unified')
    return fig

# Основное приложение
st.title("🏪 Система предсказания продаж")
uploaded_file = st.file_uploader("Загрузите Excel файл", type=['xlsx', 'xls'])

if uploaded_file:
    df = load_and_validate_data(uploaded_file)
    if df is not None:
        show_data_statistics(df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_magazin = st.selectbox("Выберите магазин", ['Все'] + list(df['Magazin'].unique()))
        with col2:
            selected_segment = st.selectbox("Выберите сегмент", ['Все'] + list(df['Segment'].unique()))
        with col3:
            model_type = st.selectbox("Модель прогнозирования", ['Prophet', 'CatBoost'])
        with col4:
            forecast_days = st.selectbox("Период прогноза", [7, 14, 30])
        
        if st.button("🔮 Показать прогноз", type="primary"):
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
                
                show_forecast_statistics(filtered_df, forecast, forecast_days, selected_magazin, selected_segment, model_type, df)
                
                st.subheader("📈 Результаты модели")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Модель", model_type)
                with col2:
                    st.metric("Тип модели", "Машинное обучение" if model_type == 'CatBoost' else "Временные ряды")
                
                st.subheader("📊 Прогноз продаж")
                fig = plot_forecast(prophet_data, forecast, model_type, f"Прогноз продаж - {model_type}")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🔍 Анализ модели")
                if model_type == 'Prophet':
                    # Компоненты Prophet
                    fig_components = plot_prophet_components(model, forecast)
                    st.plotly_chart(fig_components, use_container_width=True)
                    
                    # Сезонность Prophet
                    fig_seasonality = plot_prophet_seasonality(forecast)
                    st.plotly_chart(fig_seasonality, use_container_width=True)
                    
                    # Тренд
                    fig_trend = px.line(forecast, x='ds', y='trend', title='Тренд продаж')
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                # График продаж в деньгах по месяцам
                fig_monthly = plot_monthly_revenue_by_segment(df, selected_magazin, selected_segment)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                st.subheader("🏆 Топ-10 моделей по сегментам")
                segments_top_models = get_top_models_by_segment(df, selected_magazin)
                for segment, top_models in segments_top_models.items():
                    with st.expander(f"📦 Сегмент: {segment}"):
                        if not top_models.empty:
                            st.dataframe(
                                top_models[['Model', 'Qty', 'Sum', 'Price']].rename(columns={
                                    'Model': 'Модель', 'Qty': 'Количество', 'Sum': 'Сумма', 'Price': 'Средняя цена'
                                }), hide_index=True)
                        else:
                            st.info("Нет данных для этого сегмента")
                
                st.subheader("💡 Рекомендации для магазина")
                magazin_name = selected_magazin if selected_magazin != 'Все' else 'всех магазинов'
                st.markdown(f"**Анализ для: {magazin_name}**")
                insights, problems = generate_insights_for_magazin(df, forecast, model_type, selected_magazin)
                if problems:
                    st.markdown("### 🚨 Выявленные проблемы:")
                    for problem in problems:
                        st.error(problem)
                st.markdown("### 📋 Рекомендации:")
                for insight in insights:
                    st.markdown(insight)
                
                st.subheader("📋 Детальный прогноз по дням")
                forecast_display = forecast.tail(forecast_days).copy()
                segment_volatility = calculate_segment_volatility(df, selected_magazin, selected_segment)
                realistic, optimistic, pessimistic = get_forecast_scenarios(forecast_display, model_type, segment_volatility)
                forecast_display = pd.DataFrame({
                    'Дата': pd.to_datetime(forecast_display['ds']).dt.date,
                    'Пессимистичный': pessimistic.round(0).astype(int),
                    'Реальный': realistic.round(0).astype(int),
                    'Оптимистичный': optimistic.round(0).astype(int)
                })
                st.dataframe(forecast_display, hide_index=True)

else:
    st.info("👆 Загрузите Excel файл с данными о продажах для начала работы")
    st.subheader("📋 Требуемый формат данных")
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
    - **Prophet** - статистическая модель для анализа временных рядов с автоматическим определением сезонности
    - **CatBoost** - модель машинного обучения для более точного прогнозирования с учетом множественных факторов
    """)