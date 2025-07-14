import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置全局字体，确保中文能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 针对Windows系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300


class WeatherSpider:
    """天气数据爬虫类，用于爬取历史天气信息"""

    def __init__(self):
        # 初始化请求头生成器和会话
        self.ua = UserAgent()
        self.session = requests.Session()

        # 爬虫配置参数
        self.retry_count = 3  # 请求失败时的重试次数
        self.delay_range = (1, 3)  # 请求间隔的随机延迟范围(秒)

    def generate_random_headers(self):
        """生成随机请求头，模拟不同浏览器访问"""
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.tianqihoubao.com/",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }

    def get_month_urls(self, start_year=2025, end_year=2025, start_month=1, end_month=6):
        """
        生成指定年份范围内的所有月份URL
        返回: 包含(year, month, url)的列表
        """
        urls = []
        for year in range(start_year, end_year + 1):
            for month in range(start_month, end_month + 1):
                url = f"https://www.tianqihoubao.com/lishi/dalian/month/{year}{month:02d}.html"
                urls.append((year, month, url))
        return urls

    def parse_date(self, date_text):
        """解析日期字符串，返回格式化的日期"""
        date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_text)
        if not date_match:
            return None
        try:
            year = date_match.group(1)
            month = int(date_match.group(2))
            day = int(date_match.group(3))
            return f"{year}-{month:02d}-{day:02d}"
        except:
            return None

    def parse_temperature(self, temp_text):
        """解析温度字符串，返回最高温和最低温"""
        temps = re.findall(r'-?\d+', temp_text)
        try:
            high_temp = int(temps[0]) if len(temps) > 0 else None
            low_temp = int(temps[1]) if len(temps) > 1 else None
            return high_temp, low_temp
        except:
            return None, None

    def extract_wind_level(self, wind_str):
        """从风力描述中提取风力等级"""
        match = re.search(r'(\d+-\d+)级', wind_str)
        if match:
            return match.group(1)
        match = re.search(r'(\d+)级', wind_str)
        if match:
            return match.group(1)
        return "0"

    def parse_weather_data(self, html_content):
        """解析HTML内容中的天气数据"""
        soup = BeautifulSoup(html_content, 'html.parser')
        weather_table = soup.find('table', class_='weather-table')
        if not weather_table:
            print("未找到天气表格")
            return []
        weather_data = []
        rows = weather_table.find_all('tr')[1:]
        for row in rows:
            if not row.find_all('td') or row.find('hr'):
                continue
            columns = row.find_all('td')
            if len(columns) < 4:
                continue
            date_element = columns[0].find('a')
            date_text = date_element.get_text(strip=True) if date_element else columns[0].get_text(strip=True)
            formatted_date = self.parse_date(date_text)
            if not formatted_date:
                continue
            weather_text = columns[1].get_text(strip=True).replace('\xa0', ' ')
            weather_parts = weather_text.split('/')
            day_weather = weather_parts[0].strip() if len(weather_parts) > 0 else ''
            night_weather = weather_parts[1].strip() if len(weather_parts) > 1 else day_weather
            high_temp, low_temp = self.parse_temperature(columns[2].get_text(strip=True))
            wind_text = columns[3].get_text(strip=True).replace('\xa0', ' ')
            wind_parts = wind_text.split('/')
            day_wind = wind_parts[0].strip() if len(wind_parts) > 0 else ''
            night_wind = wind_parts[1].strip() if len(wind_parts) > 1 else day_wind
            day_wind_level = self.extract_wind_level(day_wind)
            night_wind_level = self.extract_wind_level(night_wind)
            weather_data.append({
                '日期': formatted_date,
                '最高温度': high_temp,
                '最低温度': low_temp,
                '白天天气': day_weather,
                '夜晚天气': night_weather,
                '白天风力': day_wind_level,
                '夜晚风力': night_wind_level
            })
        return weather_data

    def fetch_month_data(self, year, month, url):
        """
        爬取指定月份的天气数据
        返回: 包含天气数据的列表
        """
        for attempt in range(self.retry_count):
            try:
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)
                print(f"正在爬取 {year}年{month}月 数据...")
                headers = self.generate_random_headers()
                response = self.session.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    print(f"请求失败，状态码: {response.status_code}, 尝试 {attempt + 1}/{self.retry_count}")
                    continue
                for encoding in ['gbk', 'gb18030', 'utf-8']:
                    try:
                        response.encoding = encoding
                        month_data = self.parse_weather_data(response.text)
                        if month_data:
                            print(f"成功爬取 {year}年{month}月 {len(month_data)} 条记录")
                            return month_data
                    except Exception as e:
                        print(f"编码 {encoding} 解析失败: {str(e)}")
                print(f"所有编码尝试失败，跳过 {year}年{month}月")
                return []
            except requests.exceptions.RequestException as e:
                print(f"网络错误: {str(e)}, 尝试 {attempt + 1}/{self.retry_count}")
            except Exception as e:
                print(f"解析错误: {str(e)}, 尝试 {attempt + 1}/{self.retry_count}")
        print(f"爬取 {year}年{month}月 数据失败，跳过")
        return []

    def crawl_weather_data(self, start_year=2025, end_year=2025, start_month=1, end_month=6):
        """
        爬取指定年份范围内的所有天气数据
        返回: 包含所有天气数据的DataFrame
        """
        month_urls = self.get_month_urls(start_year, end_year, start_month, end_month)
        print(f"准备爬取 {len(month_urls)} 个月份的数据...")
        all_data = []
        success_count = 0
        fail_count = 0
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for year, month, url in month_urls:
                futures.append(executor.submit(self.fetch_month_data, year, month, url))
            for future in futures:
                month_data = future.result()
                if month_data:
                    all_data.extend(month_data)
                    success_count += 1
                else:
                    fail_count += 1
        print(f"爬取完成: 成功 {success_count} 个月份, 失败 {fail_count} 个月份")
        if not all_data:
            print("未爬取到任何数据")
            return pd.DataFrame()
        df = pd.DataFrame(all_data)
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])
        df = df.sort_values('日期')
        df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
        return df


# 主程序
if __name__ == "__main__":
    # ====================== 第一部分：数据爬取 ======================
    print("=" * 50)
    print("开始爬取2025年1-6月天气数据...")
    spider = WeatherSpider()
    weather_2025 = spider.crawl_weather_data()

    if not weather_2025.empty:
        csv_path = "大连2025上半年天气.csv"
        weather_2025.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"数据已成功保存到 {csv_path}")

    # ====================== 第二部分：数据加载与特征工程 ======================
    print("\n" + "=" * 50)
    print("开始加载历史数据并构建预测模型...")

    # 加载2022-2024年历史数据
    try:
        data = pd.read_csv("大连天气数据2022_2024.csv")
        print(f"成功加载历史数据，共 {len(data)} 条记录")
    except Exception as e:
        print(f"加载历史数据失败: {str(e)}")
        exit()

    # 数据预处理
    data['日期'] = pd.to_datetime(data['日期'])
    data = data.dropna(subset=['最高温度', '最低温度'])

    # 特征工程
    data['月份'] = data['日期'].dt.month
    data['年份'] = data['日期'].dt.year
    data['日'] = data['日期'].dt.day
    data['季节'] = data['月份'] % 12 // 3 + 1  # 1-冬季,2-春季,3-夏季,4-秋季
    data['年序数'] = (data['年份'] - 2022) * 12 + data['月份'] - 1  # 创建连续时间序列
    data['周几'] = data['日期'].dt.dayofweek

    # 添加滞后特征
    data['最高温度_lag1'] = data['最高温度'].shift(1)
    data['最高温度_lag7'] = data['最高温度'].shift(7)
    data['最高温度_lag30'] = data['最高温度'].shift(30)

    # 填充滞后特征的缺失值
    data = data.bfill()

    # 准备训练集
    X = data[['年序数', '月份', '季节', '日', '周几', '最高温度_lag1', '最高温度_lag7', '最高温度_lag30']]
    y = data['最高温度']

    # ====================== 第三部分：模型训练与优化 ======================
    print("\n" + "=" * 50)
    print("开始训练和优化预测模型...")

    # 创建模型管道
    model_pipeline = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    )

    # 参数网格
    param_grid = {
        'randomforestregressor__n_estimators': [100, 150, 200],
        'randomforestregressor__max_depth': [None, 10, 20],
        'randomforestregressor__min_samples_split': [2, 5, 10]
    }

    # 时序交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    # 最佳模型
    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {grid_search.best_params_}")
    print(f"最佳模型分数: {-grid_search.best_score_}")

    # ====================== 第四部分：预测2025年数据 ======================
    print("\n" + "=" * 50)
    print("开始预测2025年天气数据...")

    # 准备2025年数据
    weather_2025['日期'] = pd.to_datetime(weather_2025['日期'])
    weather_2025['月份'] = weather_2025['日期'].dt.month
    weather_2025['年份'] = 2025
    weather_2025['日'] = weather_2025['日期'].dt.day
    weather_2025['季节'] = weather_2025['月份'] % 12 // 3 + 1
    weather_2025['年序数'] = (2025 - 2022) * 12 + weather_2025['月份'] - 1
    weather_2025['周几'] = weather_2025['日期'].dt.dayofweek

    # 使用历史数据初始化滞后特征
    last_date = data['日期'].max()
    weather_2025['最高温度_lag1'] = np.nan
    weather_2025['最高温度_lag7'] = np.nan
    weather_2025['最高温度_lag30'] = np.nan

    # 为2025年数据填充滞后特征
    for i, row in weather_2025.iterrows():
        current_date = row['日期']

        # 滞后1天
        lag1_date = current_date - timedelta(days=1)
        lag1_value = data[data['日期'] == lag1_date]['最高温度']
        if not lag1_value.empty:
            weather_2025.at[i, '最高温度_lag1'] = lag1_value.values[0]

        # 滞后7天
        lag7_date = current_date - timedelta(days=7)
        lag7_value = data[data['日期'] == lag7_date]['最高温度']
        if not lag7_value.empty:
            weather_2025.at[i, '最高温度_lag7'] = lag7_value.values[0]

        # 滞后30天
        lag30_date = current_date - timedelta(days=30)
        lag30_value = data[data['日期'] == lag30_date]['最高温度']
        if not lag30_value.empty:
            weather_2025.at[i, '最高温度_lag30'] = lag30_value.values[0]

    # 填充剩余的缺失值
    weather_2025 = weather_2025.bfill().ffill()

    # 预测每日最高温度
    X_pred = weather_2025[['年序数', '月份', '季节', '日', '周几', '最高温度_lag1', '最高温度_lag7', '最高温度_lag30']]
    weather_2025['预测最高温度'] = best_model.predict(X_pred)

    # 计算月度均值
    monthly_pred = weather_2025.groupby('月份')['预测最高温度'].mean().reset_index()
    monthly_real = weather_2025.groupby('月份')['最高温度'].mean().reset_index()

    # 计算预测误差
    mae = mean_absolute_error(weather_2025['最高温度'], weather_2025['预测最高温度'])
    rmse = np.sqrt(mean_squared_error(weather_2025['最高温度'], weather_2025['预测最高温度']))
    print(f"预测误差 - MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C")

    # ====================== 第五部分：可视化结果 ======================
    # ====================== 第五部分：可视化结果 ======================
    print("\n" + "=" * 50)
    print("生成预测结果可视化...")

    # 创建双Y轴图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # 顶部图表：温度对比
    sns.lineplot(x='日期', y='最高温度', data=weather_2025, ax=ax1,
                 label='真实最高温度', linewidth=1.5, marker='o')
    sns.lineplot(x='日期', y='预测最高温度', data=weather_2025, ax=ax1,
                 label='预测最高温度', linewidth=1.5, marker='x')
    ax1.set_title('2025年1-6月每日温度预测对比', fontsize=15)
    ax1.set_ylabel('温度 (°C)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 添加误差带
    ax1.fill_between(weather_2025['日期'],
                     weather_2025['预测最高温度'] - mae,
                     weather_2025['预测最高温度'] + mae,
                     color='gray', alpha=0.2, label='±MAE误差带')

    # 底部图表：误差分析 - 使用折线图替代条形图
    weather_2025['误差'] = weather_2025['预测最高温度'] - weather_2025['最高温度']
    sns.lineplot(x='日期', y='误差', data=weather_2025, ax=ax2,
                 color='blue', marker='o', label='预测误差')
    ax2.axhline(0, color='red', linestyle='--', label='零误差线')
    ax2.set_title('每日预测误差', fontsize=15)
    ax2.set_ylabel('预测误差 (°C)')
    ax2.set_xlabel('日期')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("改进预测与真实温度对比.png")
    print("预测与真实温度对比图已保存至 改进预测与真实温度对比.png")

    # 月度对比图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='月份', y='最高温度', data=monthly_real, color='skyblue', label='真实月均温度')
    sns.barplot(x='月份', y='预测最高温度', data=monthly_pred, color='salmon', alpha=0.6, label='预测月均温度')

    # 添加数据标签
    for i, row in monthly_real.iterrows():
        plt.text(i, row['最高温度'] + 0.3, f"{row['最高温度']:.1f}°C",
                 ha='center', fontsize=9)
    for i, row in monthly_pred.iterrows():
        plt.text(i, row['预测最高温度'] - 0.5, f"{row['预测最高温度']:.1f}°C",
                 ha='center', fontsize=9, color='darkred')

    plt.title('2025年1-6月平均最高温度预测对比', fontsize=15)
    plt.xlabel('月份')
    plt.ylabel('温度 (°C)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("月度预测对比.png")
    print("月度预测对比图已保存至 月度预测对比.png")

    # 模型特征重要性分析
    rf_model = best_model.named_steps['randomforestregressor']
    feature_importances = rf_model.feature_importances_

    # 获取多项式特征名称
    poly = best_model.named_steps['polynomialfeatures']
    poly_feature_names = poly.get_feature_names_out(X.columns)

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': poly_feature_names,
        '重要性': feature_importances
    }).sort_values('重要性', ascending=False)

    # 只显示前20个最重要的特征
    importance_df = importance_df.head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='重要性', y='特征', data=importance_df, palette='viridis')
    plt.title('预测模型特征重要性 (前20个)', fontsize=15)
    plt.xlabel('重要性分数')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig("特征重要性.png")
    print("特征重要性图已保存至 特征重要性.png")

    print("\n预测分析完成！")











####################
# 构建机器学习管道
model_pipeline = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
)

# 参数网格搜索
param_grid = {
    'randomforestregressor__n_estimators': [100, 150, 200],
    'randomforestregressor__max_depth': [None, 10, 20],
    'randomforestregressor__min_samples_split': [2, 5, 10]
}

# 时序交叉验证
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X, y)