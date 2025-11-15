# 🏪 Система прогнозування продажів / Sales Forecasting System

Інтелектуальна система прогнозування продажів на основі машинного навчання з використанням Facebook Prophet.

*Intelligent sales forecasting system based on machine learning using Facebook Prophet.*

---

## 📋 Зміст / Table of Contents

- [Опис](#опис--description)
- [Можливості](#можливості--features)
- [Вимоги](#вимоги--requirements)
- [Встановлення](#встановлення--installation)
- [Запуск](#запуск--usage)
- [Структура даних](#структура-даних--data-structure)
- [Функціонал](#функціонал--functionality)
- [Технології](#технології--technologies)

---

## 📖 Опис / Description

**[UA]** Це веб-застосунок на основі Streamlit для прогнозування продажів товарів у роздрібних магазинах. Система використовує модель Prophet від Facebook для створення точних прогнозів на основі історичних даних продажів з можливістю налаштування параметрів та попередньої обробки даних.

**[EN]** This is a Streamlit-based web application for forecasting product sales in retail stores. The system uses Facebook's Prophet model to create accurate forecasts based on historical sales data with configurable parameters and data preprocessing options.

---

## ✨ Можливості / Features

### 📊 Джерела даних / Data Sources
- ✅ Завантаження з локальних Excel файлів / Upload from local Excel files
- ✅ Інтеграція з Google Sheets / Google Sheets integration
- ✅ Кешування даних для швидкої роботи / Data caching for fast performance

### 🔧 Попередня обробка даних / Data Preprocessing
- ✅ Видалення викидів методом IQR / Outlier removal using IQR method
- ✅ Згладжування даних (ковзне середнє, експоненційне, фільтр Савіцького-Голея) / Data smoothing (moving average, exponential, Savitzky-Golay filter)
- ✅ Автоматична валідація даних / Automatic data validation

### 📈 Прогнозування / Forecasting
- ✅ Прогноз на 7-90 днів / Forecast for 7-90 days
- ✅ Три сценарії прогнозу: песимістичний, реалістичний, оптимістичний / Three forecast scenarios: pessimistic, realistic, optimistic
- ✅ Аналіз компонентів прогнозу (тренд, сезонність) / Forecast components analysis (trend, seasonality)
- ✅ Розрахунок метрик точності (MAE, RMSE, MAPE, R²) / Accuracy metrics calculation (MAE, RMSE, MAPE, R²)

### 📊 Візуалізація / Visualization
- ✅ Інтерактивні графіки з Plotly / Interactive charts with Plotly
- ✅ Місячний аналіз з прогнозом / Monthly analysis with forecast
- ✅ Аналіз топ-моделей товарів / Top product models analysis
- ✅ Візуалізація компонентів моделі / Model components visualization

### 💡 Аналітика / Analytics
- ✅ Автоматична генерація інсайтів / Automatic insights generation
- ✅ Статистика продажів / Sales statistics
- ✅ Аналіз волатильності / Volatility analysis
- ✅ Розрахунок прогнозованої виручки / Forecasted revenue calculation

### 📄 Звіти / Reports
- ✅ Експорт детального прогнозу в Excel / Export detailed forecast to Excel
- ✅ Генерація звітів у форматі Word / Generate reports in Word format
- ✅ Таблиці з метриками та рекомендаціями / Tables with metrics and recommendations

---

## 🔧 Вимоги / Requirements

- Python 3.8+
- Пакети Python (встановлюються через requirements.txt):
  - streamlit
  - pandas
  - numpy
  - plotly
  - prophet
  - catboost
  - openpyxl
  - scikit-learn

---

## 💻 Встановлення / Installation

### 1. Клонування репозиторію / Clone the repository

```bash
git clone <repository-url>
cd predict_sales
```

### 2. Створення віртуального середовища / Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Встановлення залежностей / Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Запуск / Usage

### Запуск застосунку / Run the application

```bash
streamlit run app.py
```

Застосунок відкриється у браузері за адресою `http://localhost:8501`

*The application will open in your browser at `http://localhost:8501`*

### Основний робочий процес / Main Workflow

1. **Завантаження даних / Load Data**
   - Оберіть джерело даних (локальний файл або Google Sheets)
   - Завантажте Excel файл або вкажіть URL Google Sheets

2. **Налаштування параметрів / Configure Parameters**
   - Період прогнозу (7-90 днів)
   - Методи попередньої обробки даних
   - Параметри згладжування

3. **Вибір аналізу / Select Analysis**
   - Оберіть магазин
   - Оберіть сегмент товарів

4. **Створення прогнозу / Generate Forecast**
   - Натисніть кнопку "Створити прогноз"
   - Переглядайте результати та візуалізації
   - Експортуйте звіти

---

## 📋 Структура даних / Data Structure

### Обов'язкові колонки у файлі Excel / Required columns in Excel file:

| Колонка / Column | Опис / Description | Приклад / Example |
|------------------|-------------------|-------------------|
| `Magazin` | Назва магазину / Store name | "Магазин №1" |
| `Datasales` | Дата продажу / Sale date | "2024-01-15" |
| `Art` | Артикул товару / Product article | "ART-12345" |
| `Describe` | Опис товару / Product description | "Товар А" |
| `Model` | Модель товару / Product model | "Model X" |
| `Segment` | Сегмент товару / Product segment | "Електроніка" |
| `Price` | Ціна за одиницю / Unit price | 1500.00 |
| `Qty` | Кількість проданих одиниць / Quantity sold | 5 |
| `Sum` | Загальна сума продажу / Total sale amount | 7500.00 |

### Формат дати / Date Format

Дати повинні бути в форматі ISO (YYYY-MM-DD) або стандартному форматі Excel.

*Dates should be in ISO format (YYYY-MM-DD) or standard Excel format.*

---

## 🎯 Функціонал / Functionality

### 1. Завантаження та валідація даних / Data Loading and Validation
- Перевірка наявності обов'язкових колонок
- Автоматичне перетворення типів даних
- Обробка помилок при завантаженні

### 2. Статистика даних / Data Statistics
- Загальна кількість записів
- Період даних
- Кількість магазинів та сегментів
- Загальні продажі та виручка

### 3. Попередня обробка / Preprocessing
- **Видалення викидів / Outlier Removal**: Метод IQR для видалення аномальних значень
- **Згладжування / Smoothing**:
  - MA (Moving Average) - ковзне середнє
  - EMA (Exponential Moving Average) - експоненційне згладжування
  - Savitzky-Golay - фільтр Савіцького-Голея

### 4. Навчання моделі / Model Training
- Використання Prophet для прогнозування часових рядів
- Автоматичне визначення трендів та сезонності
- Розрахунок метрик точності моделі:
  - **MAE** (Mean Absolute Error) - середня абсолютна помилка
  - **RMSE** (Root Mean Squared Error) - корінь з середньої квадратичної помилки
  - **MAPE** (Mean Absolute Percentage Error) - середня абсолютна процентна помилка
  - **R²** (R-squared) - коефіцієнт детермінації

### 5. Сценарії прогнозу / Forecast Scenarios
- 😰 **Песимістичний / Pessimistic**: Нижня межа довірчого інтервалу
- 🎯 **Реалістичний / Realistic**: Базовий прогноз моделі
- 🚀 **Оптимістичний / Optimistic**: Верхня межа довірчого інтервалу

### 6. Візуалізація / Visualization
- Графік прогнозу з історичними даними
- Компоненти моделі (тренд, тижнева/річна сезонність)
- Місячний аналіз продажів
- Топ-10 моделей товарів

### 7. Інсайти / Insights
Система автоматично генерує рекомендації на основі:
- Загального прогнозу продажів
- Середнього денного прогнозу
- Тенденції (зростання/спад)
- Прогнозованої виручки
- Мінімальних та максимальних значень

### 8. Експорт / Export
- **Excel**: Детальний прогноз з усіма сценаріями
- **Word**: Повний звіт з метриками, графіками та рекомендаціями

---

## 🛠 Технології / Technologies

- **Frontend/UI**: Streamlit
- **Прогнозування / Forecasting**: Facebook Prophet
- **Обробка даних / Data Processing**: Pandas, NumPy
- **Візуалізація / Visualization**: Plotly
- **Статистика / Statistics**: SciPy, scikit-learn
- **Експорт / Export**: openpyxl, python-docx

---

## 📊 Приклад використання / Example Usage

### Сценарій 1: Прогноз для конкретного магазину
```
1. Завантажте файл з даними продажів
2. Оберіть магазин "Магазин №1"
3. Оберіть сегмент "Електроніка"
4. Встановіть період прогнозу: 30 днів
5. Увімкніть видалення викидів
6. Оберіть метод згладжування: "Ковзне середнє"
7. Натисніть "Створити прогноз"
8. Експортуйте результати в Excel або Word
```

### Сценарій 2: Аналіз всіх магазинів
```
1. Завантажте дані
2. Оберіть "Всі магазини"
3. Оберіть "Всі сегменти"
4. Створіть прогноз
5. Переглядайте загальну статистику та топ-моделі
```

---

## 🔍 Метрики точності моделі / Model Accuracy Metrics

| Метрика / Metric | Опис / Description | Інтерпретація / Interpretation |
|------------------|-------------------|-------------------------------|
| **MAE** | Середня абсолютна помилка / Mean Absolute Error | Чим менше, тим краще / Lower is better |
| **RMSE** | Корінь з середньої квадратичної помилки / Root Mean Squared Error | Чим менше, тим краще / Lower is better |
| **MAPE** | Середня абсолютна процентна помилка / Mean Absolute Percentage Error | < 10% - відмінно / < 10% - excellent |
| **R²** | Коефіцієнт детермінації / R-squared | Близько до 1 - відмінно / Close to 1 - excellent |

---

## 💡 Поради / Tips

1. **Якість даних / Data Quality**: Чим більше історичних даних, тим точніший прогноз / More historical data leads to more accurate forecasts

2. **Попередня обробка / Preprocessing**:
   - Використовуйте видалення викидів для даних з аномаліями / Use outlier removal for data with anomalies
   - Згладжування допомагає при шумних даних / Smoothing helps with noisy data

3. **Період прогнозу / Forecast Period**:
   - Короткострокові прогнози (7-14 днів) - найточніші / Short-term forecasts (7-14 days) are most accurate
   - Довгострокові прогнози (60+ днів) - менш надійні / Long-term forecasts (60+ days) are less reliable

4. **Сезонність / Seasonality**:
   - Для точного визначення сезонності потрібно мінімум 2 повних цикли / At least 2 full cycles needed for accurate seasonality detection

5. **Google Sheets**:
   - Використовуйте кешування для економії часу / Use caching to save time
   - Оновлюйте кеш лише при зміні даних / Refresh cache only when data changes

---

## 📝 Ліцензія / License

Цей проект розроблено для комерційного використання у сфері роздрібної торгівлі.

*This project is developed for commercial use in retail.*

---

## 🤝 Підтримка / Support

Для питань та підтримки зверніться до розробника.

*For questions and support, contact the developer.*

---

## 🎨 Скріншоти / Screenshots

Застосунок має сучасний інтерфейс з:
- 🎨 Градієнтним дизайном
- 📊 Інтерактивними графіками
- 💳 Метрик-картками
- 📈 Детальними таблицями
- 💡 Інсайтами та рекомендаціями

*The application features a modern interface with:*
- *🎨 Gradient design*
- *📊 Interactive charts*
- *💳 Metric cards*
- *📈 Detailed tables*
- *💡 Insights and recommendations*

---

**Версія / Version**: 1.0
**Дата оновлення / Last Updated**: 2024
**Розробник / Developer**: Sales Analytics Team
