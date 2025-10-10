import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(
    page_title="🏪 Система прогнозирования продаж", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .problem-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_validate_data(uploaded_file):
    """Загружает и валидирует данные из Excel файла"""
    try:
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        
        df = pd.read_excel(uploaded_file)
        progress_bar.progress(50)
        
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Отсутствуют обязательные колонки: {missing_cols}")
            return None
            
        progress_bar.progress(75)
        
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Datasales']).sort_values('Datasales')
        df = df[(df['Qty'] >= 0) & (df['Price'] > 0)]
        
        progress_bar.progress(100)
        progress_bar.empty()
        
        st.success(f"✅ Данные успешно загружены! Обработано {len(df)} записей")
        return df
        
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке файла: {str(e)}")
        return None

def show_data_statistics(df):
    """Отображает статистику данных"""
    st.markdown("## 📊 Статистика данных")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""<div class="metric-container">
                <h3>📦 Всего записей</h3>
                <h2>{len(df):,}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""<div class="metric-container">
                <h3>🏷️ Уникальных товаров</h3>
                <h2>{df['Art'].nunique():,}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""<div class="metric-container">
                <h3>🏪 Магазинов</h3>
                <h2>{df['Magazin'].nunique()}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""<div class="metric-container">
                <h3>📂 Сегментов</h3>
                <h2>{df['Segment'].nunique()}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📅 **Период данных**: {df['Datasales'].min().date()} - {df['Datasales'].max().date()}")
    with col2:
        st.info(f"💰 **Общая выручка**: {df['Sum'].sum():,.0f} ГРН")
    with col3:
        st.info(f"📈 **Средние продажи/день**: {df.groupby('Datasales')['Qty'].sum().mean():.1f} шт.")

def prepare_prophet_data(df, target_col='Qty'):
    """Подготавливает данные для Prophet"""
    prophet_df = df.groupby('Datasales')[target_col].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_prophet_model(train_data, periods=30):
    """Обучает модель Prophet"""
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=0.8
        )
        
        model.add_country_holidays(country_name='UA')
        
        with st.spinner('🤖 Обучение модели Prophet...'):
            model.fit(train_data)
            
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return model, forecast
        
    except Exception as e:
        st.error(f"❌ Ошибка при обучении модели: {str(e)}")
        return None, None

def calculate_segment_volatility(df, selected_magazin, selected_segment):
    """Вычисляет волатильность сегмента"""
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

def get_forecast_scenarios(forecast, segment_volatility=0.2):
    """Создает сценарии прогноза"""
    realistic = forecast['yhat'].values
    
    if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        optimistic = forecast['yhat_upper'].values
        pessimistic = forecast['yhat_lower'].values
    else:
        pessimistic = realistic * (1 - segment_volatility)
        optimistic = realistic * (1 + segment_volatility * 0.7)
        
    return realistic, optimistic, np.maximum(pessimistic, 0)

def plot_forecast(df, forecast, title):
    """Создает интерактивный график прогноза"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['ds'], 
        y=df['y'],
        mode='lines+markers',
        name='📊 Исторические данные',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='<b>Дата:</b> %{x}<br><b>Продажи:</b> %{y}<extra></extra>'
    ))
    
    forecast_data = forecast[forecast['ds'] > df['ds'].max()]
    
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'], 
        y=forecast_data['yhat'],
        mode='lines+markers',
        name='🔮 Прогноз',
        line=dict(color='#00CC96', width=4, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>Дата:</b> %{x}<br><b>Прогноз:</b> %{y:.0f}<extra></extra>'
    ))
    
    if 'yhat_lower' in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            fill='tonexty',
            mode='lines',
            fillcolor='rgba(0, 204, 150, 0.2)',
            line_color='rgba(0,0,0,0)',
            name='📈 Доверительный интервал',
            hovertemplate='<b>Нижняя граница:</b> %{y:.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': f'<b>{title}</b>', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24, 'color': '#1f77b4'}},
        xaxis=dict(title='<b>Дата</b>', showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', tickformat='%Y-%m-%d'),
        yaxis=dict(title='<b>Количество продаж</b>', showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.8)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12)
    )
    
    return fig

def plot_prophet_components(model, forecast):
    """Создает график компонентов Prophet"""
    from prophet.plot import plot_components_plotly
    fig = plot_components_plotly(model, forecast)
    fig.update_layout(
        title_text="<b>📈 Анализ компонентов временного ряда</b>",
        title_x=0.5,
        font=dict(family="Arial", size=12),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_monthly_analysis_with_forecast(df, selected_magazin, selected_segment, model, forecast_days):
    """Создает анализ продаж по месяцам с прогнозом"""
    filtered_df = df.copy()
    if selected_magazin != 'Все':
        filtered_df = filtered_df[filtered_df['Magazin'] == selected_magazin]
    if selected_segment != 'Все':
        filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]
    
    # Исторические данные
    filtered_df['year_month'] = filtered_df['Datasales'].dt.to_period('M')
    monthly_data = filtered_df.groupby('year_month').agg({'Sum': 'sum', 'Qty': 'sum'}).reset_index()
    monthly_data['year_month'] = monthly_data['year_month'].astype(str)
    monthly_data['type'] = 'Исторические'
    
    # Прогноз
    if model is not None:
        prophet_data = prepare_prophet_data(filtered_df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast_full = model.predict(future)
        
        # Берем только будущие даты
        forecast_future = forecast_full[forecast_full['ds'] > prophet_data['ds'].max()].copy()
        forecast_future['year_month'] = pd.to_datetime(forecast_future['ds']).dt.to_period('M')
        
        # Средняя цена для расчета выручки
        avg_price = filtered_df['Price'].mean() if not filtered_df.empty else 0
        
        forecast_monthly = forecast_future.groupby('year_month').agg({'yhat': 'sum'}).reset_index()
        forecast_monthly['Sum'] = forecast_monthly['yhat'] * avg_price
        forecast_monthly['Qty'] = forecast_monthly['yhat']
        forecast_monthly['year_month'] = forecast_monthly['year_month'].astype(str)
        forecast_monthly['type'] = 'Прогноз'
        
        # Объединяем
        combined_data = pd.concat([
            monthly_data[['year_month', 'Sum', 'type']],
            forecast_monthly[['year_month', 'Sum', 'type']]
        ], ignore_index=True)
    else:
        combined_data = monthly_data[['year_month', 'Sum', 'type']]
    
    # График
    fig = go.Figure()
    
    hist_data = combined_data[combined_data['type'] == 'Исторические']
    fig.add_trace(go.Bar(
        x=hist_data['year_month'],
        y=hist_data['Sum'],
        name='💰 Историческая выручка',
        marker_color='#636EFA',
        text=hist_data['Sum'].round(0),
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>Месяц:</b> %{x}<br><b>Выручка:</b> %{y:,.0f} ГРН<extra></extra>'
    ))
    
    if model is not None:
        forecast_data = combined_data[combined_data['type'] == 'Прогноз']
        fig.add_trace(go.Bar(
            x=forecast_data['year_month'],
            y=forecast_data['Sum'],
            name='🔮 Прогноз выручки',
            marker_color='#00CC96',
            marker_pattern_shape="/",
            text=forecast_data['Sum'].round(0),
            texttemplate='%{text:,.0f}',
            textposition='outside',
            hovertemplate='<b>Месяц:</b> %{x}<br><b>Прогноз:</b> %{y:,.0f} ГРН<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'<b>📊 Анализ продаж по месяцам с прогнозом - {selected_segment}</b>',
        title_x=0.5,
        xaxis_title='<b>Месяц</b>',
        yaxis_title='<b>Выручка (ГРН)</b>',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        barmode='group'
    )
    
    return fig

def get_top_models_by_segment(df, selected_magazin):
    """Получает топ моделей по сегментам"""
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

def generate_insights(df, forecast_data, selected_magazin, selected_segment):
    """Генерирует инсайты и рекомендации"""
    insights = []
    problems = []
    
    filtered_df = df.copy()
    if selected_magazin != 'Все':
        filtered_df = filtered_df[filtered_df['Magazin'] == selected_magazin]
    if selected_segment != 'Все':
        filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]
    
    prophet_data = prepare_prophet_data(filtered_df)
    
    if len(prophet_data) >= 30:
        recent_data = prophet_data.tail(30)['y'].mean()
        older_data = prophet_data.iloc[-60:-30]['y'].mean() if len(prophet_data) >= 60 else recent_data
        
        growth_rate = (recent_data / older_data - 1) * 100 if older_data > 0 else 0
        
        if growth_rate > 10:
            insights.append(f"📈 **РОСТ ПРОДАЖ**: Наблюдается рост на {growth_rate:.1f}%. Увеличьте закупки!")
        elif growth_rate < -10:
            insights.append(f"📉 **СНИЖЕНИЕ ПРОДАЖ**: Падение на {abs(growth_rate):.1f}%. Требуются промо-акции!")
            problems.append("⚠️ **ПРОБЛЕМА**: Значительное снижение продаж")
    
    if len(prophet_data) > 1:
        volatility = prophet_data['y'].std() / prophet_data['y'].mean() if prophet_data['y'].mean() > 0 else 0
        
        if volatility > 0.5:
            insights.append("⚡ **ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ**: Увеличьте страховые запасы на 30%")
            problems.append("🎯 **НЕСТАБИЛЬНОСТЬ**: Сложно планировать из-за высокой волатильности")
        elif volatility < 0.1:
            insights.append("✅ **СТАБИЛЬНЫЕ ПРОДАЖИ**: Можно точно планировать закупки")
    
    if len(forecast_data) > 7:
        forecast_trend = forecast_data.iloc[-7:]['yhat'].mean() / forecast_data.iloc[:7]['yhat'].mean()
        
        if forecast_trend > 1.05:
            insights.append("🚀 **ПРОГНОЗ РОСТА**: Ожидается рост. Подготовьтесь к увеличению спроса!")
        elif forecast_trend < 0.95:
            insights.append("⬇️ **ПРОГНОЗ СПАДА**: Возможно снижение. Планируйте промо-активности")
    
    if not insights:
        insights.append("📊 **СТАБИЛЬНАЯ СИТУАЦИЯ**: Продолжайте мониторинг показателей")
    
    return insights, problems

def show_forecast_statistics(filtered_df, forecast, forecast_days, selected_magazin, selected_segment, df):
    """Показывает статистику прогноза"""
    st.markdown("## 📊 Статистика прогноза")
    
    historical_data = filtered_df.groupby('Datasales')['Qty'].sum().reset_index()
    
    if len(historical_data) >= forecast_days:
        hist_qty = historical_data.tail(forecast_days)['Qty'].sum()
    else:
        daily_avg = historical_data['Qty'].mean() if len(historical_data) > 0 else 0
        hist_qty = daily_avg * forecast_days
    
    avg_price = filtered_df['Price'].mean() if not filtered_df.empty else 0
    hist_sum = hist_qty * avg_price
    
    segment_volatility = calculate_segment_volatility(df, selected_magazin, selected_segment)
    forecast_period = forecast.tail(forecast_days) if len(forecast) > forecast_days else forecast
    
    realistic, optimistic, pessimistic = get_forecast_scenarios(forecast_period, segment_volatility)
    
    realistic_qty = realistic.sum()
    optimistic_qty = optimistic.sum()
    pessimistic_qty = pessimistic.sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Прогноз количества")
        
        scenarios_df = pd.DataFrame({
            'Сценарий': ['📊 Исторический', '😰 Пессимистичный', '🎯 Реальный', '🚀 Оптимистичный'],
            'Количество': [f"{int(hist_qty):,}", f"{int(pessimistic_qty):,}", 
                          f"{int(realistic_qty):,}", f"{int(optimistic_qty):,}"],
            'Изменение': [
                '—',
                f"{((pessimistic_qty/hist_qty - 1) * 100):+.1f}%" if hist_qty > 0 else '—',
                f"{((realistic_qty/hist_qty - 1) * 100):+.1f}%" if hist_qty > 0 else '—',
                f"{((optimistic_qty/hist_qty - 1) * 100):+.1f}%" if hist_qty > 0 else '—'
            ]
        })
        
        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 💰 Прогноз выручки")
        
        revenue_df = pd.DataFrame({
            'Сценарий': ['📊 Исторический', '😰 Пессимистичный', '🎯 Реальный', '🚀 Оптимистичный'],
            'Выручка (ГРН)': [
                f"{hist_sum:,.0f}",
                f"{pessimistic_qty * avg_price:,.0f}",
                f"{realistic_qty * avg_price:,.0f}",
                f"{optimistic_qty * avg_price:,.0f}"
            ]
        })
        
        st.dataframe(revenue_df, use_container_width=True, hide_index=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📊 **Волатильность**: {segment_volatility:.1%}")
    with col2:
        st.info(f"💰 **Средняя цена**: {avg_price:.0f} ГРН")
    with col3:
        st.info(f"📅 **Период прогноза**: {forecast_days} дней")

def main():
    st.markdown('<h1 class="main-header">🏪 Система прогнозирования продаж</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ Настройки")
        
        uploaded_file = st.file_uploader(
            "📁 Загрузите файл с данными", 
            type=['xlsx', 'xls'],
            help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
        )
    
    if uploaded_file is None:
        st.markdown("""
        ## 👋 Добро пожаловать в систему прогнозирования продаж!
        
        ### 📋 Инструкция по использованию:
        
        1. **Загрузите файл Excel** с данными о продажах через боковую панель
        2. **Выберите параметры** для анализа (магазин, сегмент, период)
        3. **Получите прогноз** с помощью модели Prophet
        4. **Изучите рекомендации** для оптимизации продаж
        
        ### 📊 Требования к данным:
        """)
        
        required_columns = pd.DataFrame({
            'Колонка': ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum'],
            'Описание': [
                'Название магазина',
                'Дата продажи',
                'Артикул товара',
                'Описание товара',
                'Модель товара',
                'Сегмент товара',
                'Цена за единицу',
                'Количество',
                'Общая сумма'
            ],
            'Тип данных': ['Текст', 'Дата', 'Текст', 'Текст', 'Текст', 'Текст', 'Число', 'Число', 'Число']
        })
        
        st.dataframe(required_columns, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### 🚀 Возможности системы:
        - 📈 **Прогнозирование** продаж на 7, 14 или 30 дней
        - 🎯 **Три сценария** прогноза (пессимистичный, реальный, оптимистичный)
        - 📊 **Анализ трендов** и сезонности
        - 🏆 **Топ товаров** по сегментам
        - 💡 **Умные рекомендации** для бизнеса
        """)
        
        return
    
    df = load_and_validate_data(uploaded_file)
    
    if df is None:
        return
    
    show_data_statistics(df)
    
    st.markdown("## 🎛️ Панель управления")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_magazin = st.selectbox(
            "🏪 Выберите магазин",
            ['Все'] + sorted(df['Magazin'].unique().tolist()),
            help="Выберите конкретный магазин или 'Все' для общего анализа"
        )
    
    with col2:
        selected_segment = st.selectbox(
            "📦 Выберите сегмент",
            ['Все'] + sorted(df['Segment'].unique().tolist()),
            help="Выберите сегмент товаров для анализа"
        )
    
    with col3:
        forecast_days = st.selectbox(
            "📅 Период прогноза",
            [7, 14, 30],
            index=1,
            help="Количество дней для прогнозирования"
        )
    
    if st.button("🔮 Создать прогноз", type="primary", use_container_width=True):
        filtered_df = df.copy()
        
        if selected_magazin != 'Все':
            filtered_df = filtered_df[filtered_df['Magazin'] == selected_magazin]
        
        if selected_segment != 'Все':
            filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]
        
        if len(filtered_df) < 10:
            st.error("❌ Недостаточно данных для прогнозирования (минимум 10 записей)")
            return
        
        prophet_data = prepare_prophet_data(filtered_df)
        model, forecast = train_prophet_model(prophet_data, periods=forecast_days)
        
        if model is None or forecast is None:
            return
        
        st.success("✅ Модель успешно обучена!")
        
        show_forecast_statistics(filtered_df, forecast, forecast_days, selected_magazin, selected_segment, df)
        
        st.markdown("## 📈 Прогноз продаж")
        
        fig_main = plot_forecast(
            prophet_data, 
            forecast, 
            f"Прогноз продаж - {selected_magazin} / {selected_segment}"
        )
        st.plotly_chart(fig_main, use_container_width=True)
        
        st.markdown("## 🔍 Детальный анализ")
        
        fig_components = plot_prophet_components(model, forecast)
        st.plotly_chart(fig_components, use_container_width=True)
        
        st.markdown("## 📊 Анализ по месяцам с прогнозом выручки")
        fig_monthly = plot_monthly_analysis_with_forecast(df, selected_magazin, selected_segment, model, forecast_days)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        st.markdown("## 🏆 Топ-10 моделей по сегментам")
        
        segments_top_models = get_top_models_by_segment(df, selected_magazin)
        
        if segments_top_models:
            tabs = st.tabs([f"📦 {segment}" for segment in segments_top_models.keys()])
            
            for tab, (segment, top_models) in zip(tabs, segments_top_models.items()):
                with tab:
                    if not top_models.empty:
                        display_df = top_models[['Model', 'Qty', 'Sum', 'Price']].rename(columns={
                            'Model': '🏷️ Модель',
                            'Qty': '📦 Количество',
                            'Sum': '💰 Выручка (ГРН)',
                            'Price': '💵 Средняя цена'
                        })
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("🔍 Нет данных для этого сегмента")
        
        st.markdown("## 💡 Инсайты и рекомендации")
        
        insights, problems = generate_insights(df, forecast, selected_magazin, selected_segment)
        
        if problems:
            st.markdown("### 🚨 Выявленные проблемы:")
            for problem in problems:
                st.markdown(f'<div class="problem-card">{problem}</div>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 Рекомендации:")
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        st.markdown("## 📋 Детальный прогноз по дням")
        
        forecast_display = forecast.tail(forecast_days).copy()
        segment_volatility = calculate_segment_volatility(df, selected_magazin, selected_segment)
        
        realistic, optimistic, pessimistic = get_forecast_scenarios(forecast_display, segment_volatility)
        
        detailed_forecast = pd.DataFrame({
            '📅 Дата': pd.to_datetime(forecast_display['ds']).dt.strftime('%Y-%m-%d (%A)'),
            '😰 Пессимистичный': pessimistic.round(0).astype(int),
            '🎯 Реальный': realistic.round(0).astype(int),
            '🚀 Оптимистичный': optimistic.round(0).astype(int),
            '📊 Тренд': forecast_display['trend'].round(0).astype(int)
        })
        
        st.dataframe(detailed_forecast, use_container_width=True, hide_index=True)
        
        st.markdown("## 📈 Дополнительная аналитика")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_daily_forecast = realistic.mean()
            st.metric(
                "📊 Средние продажи/день",
                f"{avg_daily_forecast:.0f}",
                delta=f"{avg_daily_forecast - prophet_data['y'].tail(30).mean():.0f}"
            )
        
        with col2:
            total_forecast = realistic.sum()
            st.metric(
                "📦 Общий прогноз",
                f"{total_forecast:.0f}",
                delta=f"{total_forecast - prophet_data['y'].tail(forecast_days).sum():.0f}"
            )
        
        with col3:
            avg_price = filtered_df['Price'].mean() if not filtered_df.empty else 0
            forecast_revenue = total_forecast * avg_price
            st.metric(
                "💰 Прогноз выручки",
                f"{forecast_revenue:,.0f} ГРН"
            )
        
        with col4:
            confidence_score = (1 - segment_volatility) * 100
            st.metric(
                "🎯 Уверенность прогноза",
                f"{confidence_score:.0f}%"
            )
        
        st.markdown("## 📥 Экспорт результатов")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_data = detailed_forecast.copy()
            export_data['Магазин'] = selected_magazin
            export_data['Сегмент'] = selected_segment
            
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📊 Скачать прогноз (CSV)",
                data=csv,
                file_name=f"forecast_{selected_magazin}_{selected_segment}_{forecast_days}days.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            report = f"""
# Отчет по прогнозированию продаж

**Дата создания:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Магазин:** {selected_magazin}
**Сегмент:** {selected_segment}
**Период прогноза:** {forecast_days} дней

## Основные показатели:
- **Общий прогноз:** {total_forecast:.0f} единиц
- **Средние продажи/день:** {avg_daily_forecast:.0f} единиц
- **Прогнозная выручка:** {forecast_revenue:,.0f} ГРН
- **Уверенность прогноза:** {confidence_score:.0f}%

## Рекомендации:
{chr(10).join([f"- {insight}" for insight in insights])}
            """
            
            st.download_button(
                label="📄 Скачать отчет (TXT)",
                data=report,
                file_name=f"report_{selected_magazin}_{selected_segment}.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
