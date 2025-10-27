import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.signal import savgol_filter
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Конфігурація сторінки
st.set_page_config(
    page_title="🏪 Система прогнозування продажів",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стилі
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        border-left: 5px solid #fff;
    }
    
    .problem-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 5px solid #ee5a6f;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
        font-weight: 500;
    }
    
    .accuracy-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_validate_data(uploaded_file):
    """Завантажує та валідує дані з Excel файлу"""
    try:
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        
        df = pd.read_excel(uploaded_file)
        progress_bar.progress(50)
        
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Відсутні обов'язкові колонки: {missing_cols}")
            return None
            
        progress_bar.progress(75)
        
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Datasales']).sort_values('Datasales')
        df = df[(df['Qty'] >= 0) & (df['Price'] > 0)]
        
        progress_bar.progress(100)
        progress_bar.empty()
        
        st.success(f"✅ Дані успішно завантажені! Оброблено {len(df)} записів")
        return df
        
    except Exception as e:
        st.error(f"❌ Помилка при завантаженні файлу: {str(e)}")
        return None

def show_data_statistics(df):
    """Відображає статистику даних"""
    st.markdown("## 📊 Статистика даних")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""<div class="metric-container">
                <h3>📦 Всього записів</h3>
                <h2>{len(df):,}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""<div class="metric-container">
                <h3>🏷️ Унікальних товарів</h3>
                <h2>{df['Art'].nunique():,}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""<div class="metric-container">
                <h3>🏪 Магазинів</h3>
                <h2>{df['Magazin'].nunique()}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""<div class="metric-container">
                <h3>📂 Сегментів</h3>
                <h2>{df['Segment'].nunique()}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📅 **Період даних**: {df['Datasales'].min().date()} - {df['Datasales'].max().date()}")
    with col2:
        st.info(f"💰 **Загальна виручка**: {df['Sum'].sum():,.0f} ГРН")
    with col3:
        st.info(f"📈 **Середні продажі/день**: {df.groupby('Datasales')['Qty'].sum().mean():.1f} шт.")

def remove_outliers_iqr(data, multiplier=1.5):
    """Видаляє викиди методом IQR з коректним розрахунком границь"""
    if len(data) < 4:
        return data
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Правильний розрахунок границь
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return data.clip(lower=lower_bound, upper=upper_bound)

def smooth_data(data, method='ma', window=7):
    """Згладжує дані різними методами"""
    if method == 'ma':
        return data.rolling(window=window, min_periods=1, center=True).mean()
    elif method == 'ema':
        return data.ewm(span=window, adjust=False).mean()
    elif method == 'savgol' and len(data) >= window:
        # Перевірка парності вікна для Savitzky-Golay
        if window % 2 == 0:
            window += 1
        return pd.Series(savgol_filter(data, window, 3), index=data.index)
    return data

def create_prophet_forecast(df, periods=30):
    """Створює прогноз за допомогою Prophet"""
    prophet_df = df.copy()
    prophet_df = prophet_df.rename(columns={'Datasales': 'ds', 'Qty': 'y'})
    prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
    
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast, model

def calculate_accuracy(actual, predicted):
    """Розраховує метрики точності прогнозу"""
    mask = actual > 0
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    if len(actual_filtered) == 0:
        return None
    
    mae = mean_absolute_error(actual_filtered, predicted_filtered)
    rmse = np.sqrt(mean_squared_error(actual_filtered, predicted_filtered))
    mape = mean_absolute_percentage_error(actual_filtered, predicted_filtered) * 100
    r2 = r2_score(actual_filtered, predicted_filtered)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Accuracy': max(0, 100 - mape)
    }

def create_word_report(forecast_data, magazin, segment, days, total, avg_daily, revenue, confidence, metrics, insights, df, prophet_data):
    """Створює звіт у форматі Word"""
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        doc = Document()
        
        # Заголовок
        title = doc.add_heading(f'Прогноз продажів: {magazin} / {segment}', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Основна інформація
        doc.add_heading('Основні показники', level=1)
        p = doc.add_paragraph()
        p.add_run(f'Період прогнозу: {days} днів\n').bold = True
        p.add_run(f'Прогноз продажів: {total:,.0f} шт.\n')
        p.add_run(f'Середньодобові продажі: {avg_daily:,.1f} шт.\n')
        p.add_run(f'Очікувана виручка: {revenue:,.0f} ГРН\n')
        p.add_run(f'Рівень впевненості: {confidence:.1f}%\n')
        
        # Метрики точності
        if metrics:
            doc.add_heading('Метрики точності', level=1)
            table = doc.add_table(rows=5, cols=2)
            table.style = 'Light Grid Accent 1'
            
            metrics_data = [
                ('MAE', f"{metrics['MAE']:.2f}"),
                ('RMSE', f"{metrics['RMSE']:.2f}"),
                ('MAPE', f"{metrics['MAPE']:.2f}%"),
                ('R²', f"{metrics['R2']:.4f}"),
                ('Точність', f"{metrics['Accuracy']:.2f}%")
            ]
            
            for i, (metric, value) in enumerate(metrics_data):
                table.rows[i].cells[0].text = metric
                table.rows[i].cells[1].text = value
        
        # Інсайти
        doc.add_heading('Ключові інсайти', level=1)
        for insight in insights:
            doc.add_paragraph(insight, style='List Bullet')
        
        # Збереження в BytesIO
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        return None

def main():
    st.markdown('<h1 class="main-header">🏪 Система прогнозування продажів</h1>', unsafe_allow_html=True)
    
    # Бічна панель
    with st.sidebar:
        st.header("⚙️ Налаштування")
        
        uploaded_file = st.file_uploader(
            "📤 Завантажте Excel файл",
            type=['xlsx', 'xls'],
            help="Файл повинен містити колонки: Magazin, Datasales, Art, Model, Segment, Price, Qty, Sum"
        )
        
        if uploaded_file:
            st.success("✅ Файл завантажено!")
    
    if not uploaded_file:
        st.info("👆 Завантажте Excel файл для початку роботи")
        
        st.markdown("""
        ### 📋 Вимоги до файлу:
        - Формат: Excel (.xlsx, .xls)
        - Обов'язкові колонки:
          - `Magazin` - назва магазину
          - `Datasales` - дата продажу
          - `Art` - артикул товару
          - `Model` - модель товару
          - `Segment` - сегмент товару
          - `Price` - ціна
          - `Qty` - кількість
          - `Sum` - сума
        """)
        return
    
    # Завантаження даних
    df = load_and_validate_data(uploaded_file)
    
    if df is None:
        return
    
    # Статистика даних
    show_data_statistics(df)
    
    st.markdown("---")
    
    # Фільтри
    st.markdown("## 🎯 Параметри прогнозу")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        magazines = ['Всі магазини'] + sorted(df['Magazin'].unique().tolist())
        selected_magazin = st.selectbox("🏪 Оберіть магазин", magazines)
    
    with col2:
        segments = ['Всі сегменти'] + sorted(df['Segment'].unique().tolist())
        selected_segment = st.selectbox("📂 Оберіть сегмент", segments)
    
    with col3:
        forecast_days = st.slider("📅 Період прогнозу (днів)", 7, 90, 30)
    
    # Розширені налаштування
    with st.expander("⚙️ Розширені налаштування"):
        col1, col2 = st.columns(2)
        
        with col1:
            remove_outliers = st.checkbox("🎯 Видалити викиди", value=True)
            outlier_multiplier = st.slider("Множник IQR", 1.0, 3.0, 1.5, 0.1) if remove_outliers else 1.5
        
        with col2:
            smooth_method = st.selectbox(
                "📊 Метод згладжування",
                ['ma', 'ema', 'savgol'],
                format_func=lambda x: {'ma': 'Ковзне середнє', 'ema': 'Експоненційне згладжування', 'savgol': 'Savitzky-Golay'}[x]
            )
            smooth_window = st.slider("Вікно згладжування", 3, 21, 7, 2)
    
    # Фільтрація даних
    filtered_df = df.copy()
    
    if selected_magazin != 'Всі магазини':
        filtered_df = filtered_df[filtered_df['Magazin'] == selected_magazin]
    
    if selected_segment != 'Всі сегменти':
        filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]
    
    if len(filtered_df) == 0:
        st.warning("⚠️ Немає даних для обраних параметрів")
        return
    
    # Агрегація даних по датах
    daily_sales = filtered_df.groupby('Datasales').agg({
        'Qty': 'sum',
        'Sum': 'sum'
    }).reset_index()
    
    # Обробка викидів та згладжування
    if remove_outliers:
        daily_sales['Qty'] = remove_outliers_iqr(daily_sales['Qty'], outlier_multiplier)
    
    daily_sales['Qty_smooth'] = smooth_data(daily_sales['Qty'], smooth_method, smooth_window)
    
    # Створення прогнозу Prophet
    forecast, prophet_model = create_prophet_forecast(daily_sales, forecast_days)
    
    # Розрахунок метрик
    historical_dates = daily_sales['Datasales']
    forecast_historical = forecast[forecast['ds'].isin(historical_dates)]
    
    if len(forecast_historical) > 0:
        merged = daily_sales.merge(forecast_historical[['ds', 'yhat']], left_on='Datasales', right_on='ds')
        accuracy_metrics = calculate_accuracy(merged['Qty'], merged['yhat'])
    else:
        accuracy_metrics = None
    
    # Прогнозні значення
    future_forecast = forecast[forecast['ds'] > daily_sales['Datasales'].max()].head(forecast_days)
    
    total_forecast = future_forecast['yhat'].sum()
    avg_daily_forecast = future_forecast['yhat'].mean()
    avg_price = filtered_df['Price'].mean()
    forecast_revenue = total_forecast * avg_price
    
    # Рівень впевненості
    if accuracy_metrics:
        confidence_score = accuracy_metrics['Accuracy']
    else:
        confidence_score = 85.0
    
    # Візуалізація прогнозу
    st.markdown("## 📈 Прогноз продажів")
    
    fig = go.Figure()
    
    # Історичні дані
    fig.add_trace(go.Scatter(
        x=daily_sales['Datasales'],
        y=daily_sales['Qty'],
        name='Фактичні продажі',
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    
    # Прогноз
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat'],
        name='Прогноз',
        mode='lines',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Довірчий інтервал
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_upper'],
        name='Верхня межа',
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_lower'],
        name='Довірчий інтервал',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.2)',
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=f"Прогноз продажів на {forecast_days} днів",
        xaxis_title="Дата",
        yaxis_title="Кількість (шт.)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Результати прогнозу
    st.markdown("## 📊 Результати прогнозу")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""<div class="metric-container">
                <h3>📦 Прогноз продажів</h3>
                <h2>{total_forecast:,.0f} шт.</h2>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""<div class="metric-container">
                <h3>📈 Середньодобові</h3>
                <h2>{avg_daily_forecast:,.1f} шт.</h2>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""<div class="metric-container">
                <h3>💰 Очікувана виручка</h3>
                <h2>{forecast_revenue:,.0f} ГРН</h2>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""<div class="metric-container">
                <h3>🎯 Впевненість</h3>
                <h2>{confidence_score:.1f}%</h2>
            </div>""",
            unsafe_allow_html=True
        )
    
    # Метрики точності
    if accuracy_metrics:
        st.markdown("### 📏 Метрики точності моделі")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_display = [
            ("MAE", accuracy_metrics['MAE'], "📊"),
            ("RMSE", accuracy_metrics['RMSE'], "📈"),
            ("MAPE", f"{accuracy_metrics['MAPE']:.2f}%", "🎯"),
            ("R²", accuracy_metrics['R2'], "📐")
        ]
        
        cols = [col1, col2, col3, col4]
        for col, (name, value, emoji) in zip(cols, metrics_display):
            with col:
                if name == "MAPE":
                    st.metric(f"{emoji} {name}", value)
                elif name == "R²":
                    st.metric(f"{emoji} {name}", f"{value:.4f}")
                else:
                    st.metric(f"{emoji} {name}", f"{value:.2f}")
    
    # Аналіз трендів
    st.markdown("## 📉 Аналіз трендів")
    
    with st.expander("🔍 Порівняння періодів"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Оберіть періоди для порівняння")
            
            min_date = df['Datasales'].min()
            max_date = df['Datasales'].max()
            
            period1_start = st.date_input("Початок 1-го періоду", min_date)
            period1_end = st.date_input("Кінець 1-го періоду", min_date + pd.Timedelta(days=30))
        
        with col2:
            st.markdown("#### ")
            st.write("")
            period2_start = st.date_input("Початок 2-го періоду", max_date - pd.Timedelta(days=30))
            period2_end = st.date_input("Кінець 2-го періоду", max_date)
        
        if st.button("🔍 Порівняти періоди", use_container_width=True):
            # Фільтрація даних по періодах
            period1_data = filtered_df[
                (filtered_df['Datasales'] >= pd.Timestamp(period1_start)) &
                (filtered_df['Datasales'] <= pd.Timestamp(period1_end))
            ]
            
            period2_data = filtered_df[
                (filtered_df['Datasales'] >= pd.Timestamp(period2_start)) &
                (filtered_df['Datasales'] <= pd.Timestamp(period2_end))
            ]
            
            if len(period1_data) == 0 or len(period2_data) == 0:
                st.warning("⚠️ Недостатньо даних для порівняння")
            else:
                # Агрегація по моделях
                period1_agg = period1_data.groupby('Model').agg({
                    'Qty': 'sum',
                    'Sum': 'sum'
                }).reset_index()
                period1_agg.columns = ['Model', 'Period1_Qty', 'Period1_Revenue']
                
                period2_agg = period2_data.groupby('Model').agg({
                    'Qty': 'sum',
                    'Sum': 'sum'
                }).reset_index()
                period2_agg.columns = ['Model', 'Period2_Qty', 'Period2_Revenue']
                
                # Об'єднання даних
                trend_data = period1_agg.merge(period2_agg, on='Model', how='outer').fillna(0)
                
                trend_data['Total_Qty'] = trend_data['Period1_Qty'] + trend_data['Period2_Qty']
                trend_data['Total_Revenue'] = trend_data['Period1_Revenue'] + trend_data['Period2_Revenue']
                
                # Розрахунок зміни
                trend_data['Change_%'] = ((trend_data['Period2_Qty'] - trend_data['Period1_Qty']) / 
                                          trend_data['Period1_Qty'].replace(0, 1)) * 100
                
                # Фільтр товарів з мінімальними продажами
                trend_data = trend_data[trend_data['Total_Qty'] >= 5]
                
                # ТОП-20 товарів по зростанню
                st.markdown("### 🚀 ТОП-20 моделей із найбільшим зростанням")
                
                growing_products = trend_data[trend_data['Change_%'] > 0].copy()
                
                if len(growing_products) > 0:
                    top_20_growing = growing_products.nlargest(20, 'Change_%')
                    top_20_growing['Середня_ціна'] = top_20_growing['Total_Revenue'] / top_20_growing['Total_Qty']
                    
                    # Графік
                    fig_growth = go.Figure()
                    
                    fig_growth.add_trace(go.Bar(
                        y=top_20_growing['Model'],
                        x=top_20_growing['Change_%'],
                        orientation='h',
                        marker=dict(
                            color=top_20_growing['Change_%'],
                            colorscale='Greens',
                            showscale=True,
                            colorbar=dict(title="Зростання<br>%")
                        ),
                        text=top_20_growing['Change_%'].apply(lambda x: f'+{x:.1f}%'),
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Зростання: %{x:.1f}%<br><extra></extra>'
                    ))
                    
                    fig_growth.update_layout(
                        title="ТОП-20 моделей з найбільшим зростанням продажів",
                        xaxis_title="Зміна (%)",
                        yaxis_title="Модель",
                        height=600,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig_growth, use_container_width=True, key="top20_growth")
                    
                    # Таблиця
                    st.markdown("##### 📋 Детальна інформація")
                    
                    display_growth = top_20_growing[['Model', 'Total_Qty', 'Total_Revenue', 'Period1_Qty', 'Period2_Qty', 'Change_%']].copy()
                    display_growth = display_growth.reset_index(drop=True)
                    display_growth.index = display_growth.index + 1
                    display_growth = display_growth.rename(columns={
                        'Model': '🏷️ Модель',
                        'Total_Qty': '📦 Всього продано',
                        'Total_Revenue': '💰 Виручка (ГРН)',
                        'Period1_Qty': '📊 1-й період',
                        'Period2_Qty': '📊 2-й період',
                        'Change_%': '📈 Зростання %'
                    })
                    
                    st.dataframe(
                        display_growth.style.format({
                            '📦 Всього продано': '{:,.0f}',
                            '💰 Виручка (ГРН)': '{:,.0f}',
                            '📊 1-й період': '{:,.0f}',
                            '📊 2-й період': '{:,.0f}',
                            '📈 Зростання %': '{:.1f}%'
                        }).background_gradient(subset=['📈 Зростання %'], cmap='Greens'),
                        use_container_width=True
                    )
                    
                    # Інсайти
                    st.success(f"🎯 **Лідер зростання**: {top_20_growing.iloc[0]['Model']} - зростання {top_20_growing.iloc[0]['Change_%']:.1f}%")
                    
                    insights = []
                    
                    avg_growth = top_20_growing['Change_%'].mean()
                    if avg_growth > 50:
                        insights.append("💡 **Рекомендація**: Збільште закупівлі товарів-лідерів на 30-50%")
                    
                    high_revenue_growth = top_20_growing[top_20_growing['Total_Revenue'] > top_20_growing['Total_Revenue'].median()]
                    if len(high_revenue_growth) >= 5:
                        insights.append(f"💰 **Інсайт**: {len(high_revenue_growth)} товарів із високою виручкою показують зростання")
                    
                    for insight in insights:
                        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
                    
                else:
                    st.info("ℹ️ Немає товарів із зростанням продажів")
                
                # ТОП-20 товарів із падінням
                st.markdown("### 📉 ТОП-20 моделей із найбільшим падінням")
                
                # Тільки товари з падінням
                declining_products = trend_data[trend_data['Change_%'] < 0].copy()
                
                if len(declining_products) > 0:
                    top_20_declining = declining_products.nsmallest(20, 'Change_%')
                    top_20_declining['Середня_ціна'] = top_20_declining['Total_Revenue'] / top_20_declining['Total_Qty']
                    
                    # Графік
                    fig_decline = go.Figure()
                    
                    fig_decline.add_trace(go.Bar(
                        y=top_20_declining['Model'],
                        x=top_20_declining['Change_%'],
                        orientation='h',
                        marker=dict(
                            color=top_20_declining['Change_%'],
                            colorscale='Reds_r',
                            showscale=True,
                            colorbar=dict(title="Падіння<br>%")
                        ),
                        text=top_20_declining['Change_%'].apply(lambda x: f'{x:.1f}%'),
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Падіння: %{x:.1f}%<br><extra></extra>'
                    ))
                    
                    fig_decline.update_layout(
                        title="ТОП-20 моделей з найбільшим падінням продажів",
                        xaxis_title="Зміна (%)",
                        yaxis_title="Модель",
                        height=600,
                        yaxis={'categoryorder': 'total descending'}
                    )
                    
                    st.plotly_chart(fig_decline, use_container_width=True, key="top20_decline")
                    
                    # Таблиця
                    st.markdown("##### 📋 Детальна інформація")
                    
                    display_decline = top_20_declining[['Model', 'Total_Qty', 'Total_Revenue', 'Period1_Qty', 'Period2_Qty', 'Change_%']].copy()
                    display_decline = display_decline.reset_index(drop=True)
                    display_decline.index = display_decline.index + 1
                    display_decline = display_decline.rename(columns={
                        'Model': '🏷️ Модель',
                        'Total_Qty': '📦 Всього продано',
                        'Total_Revenue': '💰 Виручка (ГРН)',
                        'Period1_Qty': '📊 1-й період',
                        'Period2_Qty': '📊 2-й період',
                        'Change_%': '📉 Падіння %'
                    })
                    
                    st.dataframe(
                        display_decline.style.format({
                            '📦 Всього продано': '{:,.0f}',
                            '💰 Виручка (ГРН)': '{:,.0f}',
                            '📊 1-й період': '{:,.0f}',
                            '📊 2-й період': '{:,.0f}',
                            '📉 Падіння %': '{:.1f}%'
                        }).background_gradient(subset=['📉 Падіння %'], cmap='Reds_r'),
                        use_container_width=True
                    )
                    
                    # Алерти та рекомендації
                    critical_decline = top_20_declining[top_20_declining['Change_%'] < -50]
                    
                    if len(critical_decline) > 0:
                        st.error(f"🚨 **КРИТИЧНО**: {len(critical_decline)} товарів з падінням більше 50%!")
                    
                    st.warning(f"⚠️ **Проблемний товар**: {top_20_declining.iloc[0]['Model']} - падіння {top_20_declining.iloc[0]['Change_%']:.1f}%")
                    
                    st.markdown("##### 💡 Рекомендації за проблемними товарами:")
                    recommendations = [
                        "🎯 Провести аналіз причин падіння (конкуренція, ціна, актуальність)",
                        "🔥 Запустити промо-акції зі знижками 20-30%",
                        "📢 Посилити маркетинг та рекламу",
                        "💡 Розглянути оновлення асортименту або заміну товару",
                        "📦 Знизити закупівлі до стабілізації ситуації"
                    ]
                    
                    for rec in recommendations:
                        st.markdown(f'<div class="insight-card">{rec}</div>', unsafe_allow_html=True)
                    
                else:
                    st.success("✅ Чудові новини! Немає товарів з падінням продажів")
                    st.balloons()
            
            # Маркетингові рекомендації
            st.markdown("### 🎯 Маркетингові рекомендації")
            
            marketing_insights = []
            
            # Аналіз топ товарів
            top_5_revenue = trend_data.nlargest(5, 'Total_Revenue')
            total_revenue_all = trend_data['Total_Revenue'].sum()
            top_5_share = (top_5_revenue['Total_Revenue'].sum() / total_revenue_all) * 100
            
            if top_5_share > 50:
                marketing_insights.append({
                    'type': 'warning',
                    'title': '⚠️ Висока концентрація',
                    'text': f"ТОП-5 товарів дають {top_5_share:.1f}% виручки. Це ризик! Диверсифікуйте портфель."
                })
            else:
                marketing_insights.append({
                    'type': 'success',
                    'title': '✅ Збалансований портфель',
                    'text': f"ТОП-5 товарів дають {top_5_share:.1f}% виручки. Гарна диверсифікація."
                })
            
            # Аналіз товарів, що падають
            declining_count = len(trend_data[trend_data['Change_%'] < -20])
            if declining_count > 10:
                marketing_insights.append({
                    'type': 'error',
                    'title': '🚨 Критична ситуація',
                    'text': f"{declining_count} товарів з падінням >20%. Терміново переглянути стратегію!"
                })
            elif declining_count > 0:
                marketing_insights.append({
                    'type': 'warning',
                    'title': '📉 Потрібна увага',
                    'text': f"{declining_count} товарів з падінням >20%. Проведіть аналіз та акції."
                })
            
            # Аналіз середнього чека (правильний розрахунок)
            avg_transaction = filtered_df['Sum'].sum() / len(filtered_df) if len(filtered_df) > 0 else 0
            median_transaction = filtered_df['Sum'].median() if len(filtered_df) > 0 else 0
            
            if avg_transaction > 0 and median_transaction > 0 and avg_transaction < median_transaction * 0.7:
                marketing_insights.append({
                    'type': 'info',
                    'title': '💳 Низький середній чек',
                    'text': f"Середній чек {avg_transaction:.0f} ГРН нижче медіани ({median_transaction:.0f} ГРН). Впровадьте cross-sell та бандли товарів."
                })
            
            # Відображення маркетингових інсайтів
            for insight in marketing_insights:
                if insight['type'] == 'success':
                    st.success(f"**{insight['title']}**: {insight['text']}")
                elif insight['type'] == 'warning':
                    st.warning(f"**{insight['title']}**: {insight['text']}")
                elif insight['type'] == 'error':
                    st.error(f"**{insight['title']}**: {insight['text']}")
                else:
                    st.info(f"**{insight['title']}**: {insight['text']}")
            
            # Конкретні дії
            st.markdown("#### 📋 План дій на найближчі 30 днів")
            
            top_20_best_count = len(trend_data.nlargest(20, 'Total_Revenue'))
            declining_count_action = len(trend_data[trend_data['Change_%'] < -20])
            
            action_plan = [
                f"1️⃣ **ТОП товари**: Збільшити бюджет на рекламу ТОП-{top_20_best_count} товарів на 30%",
                f"2️⃣ **Товари, що падають**: Провести розпродаж товарів з падінням зі знижкою 20-30%",
                f"3️⃣ **Cross-sell**: Створити 5 товарних бандлів для збільшення середнього чеку",
                f"4️⃣ **Стабільні товари**: Забезпечити постійну наявність стабільних позицій на складі",
                f"5️⃣ **Моніторинг**: Щотижня відстежувати динаміку товарів, що падають"
            ]
            
            for action in action_plan:
                st.markdown(f'<div class="insight-card">{action}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("## 📥 Експорт результатів")
            
            # Word звіт
            prophet_data = {
                'historical': daily_sales,
                'forecast': future_forecast
            }
            
            insights = [
                f"Прогноз на {forecast_days} днів: {total_forecast:,.0f} шт.",
                f"Очікувана виручка: {forecast_revenue:,.0f} ГРН",
                f"Рівень впевненості: {confidence_score:.1f}%"
            ]
            
            word_data = create_word_report(
                future_forecast, selected_magazin, selected_segment, forecast_days,
                total_forecast, avg_daily_forecast, forecast_revenue, confidence_score,
                accuracy_metrics, insights, filtered_df, prophet_data
            )
            
            if word_data:
                st.download_button(
                    label="📄 Завантажити звіт (WORD)",
                    data=word_data,
                    file_name=f"report_{selected_magazin}_{selected_segment}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            else:
                st.info("Word недоступний. Встановіть: pip install python-docx")

if __name__ == "__main__":
    main()
