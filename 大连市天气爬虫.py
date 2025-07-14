import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random
from concurrent.futures import ThreadPoolExecutor
from fake_useragent import UserAgent
from datetime import datetime


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

    def get_month_urls(self, start_year=2022, end_year=2024):
        """
        生成指定年份范围内的所有月份URL
        返回: 包含(year, month, url)的列表
        """
        urls = []
        current_year = datetime.now().year
        current_month = datetime.now().month

        # 遍历所有年份和月份
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # 跳过未来的月份
                if year > current_year or (year == current_year and month > current_month):
                    continue

                # 构建月份URL，格式为https://www.tianqihoubao.com/lishi/dalian/month/202101.html
                url = f"https://www.tianqihoubao.com/lishi/dalian/month/{year}{month:02d}.html"
                urls.append((year, month, url))

        return urls

    def parse_date(self, date_text):
        """解析日期字符串，返回格式化的日期"""
        # 使用正则表达式匹配日期格式
        date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_text)
        if not date_match:
            return None

        try:
            # 提取年、月、日并格式化为YYYY-MM-DD
            year = date_match.group(1)
            month = int(date_match.group(2))
            day = int(date_match.group(3))
            return f"{year}-{month:02d}-{day:02d}"
        except:
            return None

    def parse_temperature(self, temp_text):
        """解析温度字符串，返回最高温和最低温"""
        # 使用正则表达式提取温度数值
        temps = re.findall(r'-?\d+', temp_text)

        try:
            high_temp = int(temps[0]) if len(temps) > 0 else None
            low_temp = int(temps[1]) if len(temps) > 1 else None
            return high_temp, low_temp
        except:
            return None, None

    def extract_wind_level(self, wind_str):
        """从风力描述中提取风力等级"""
        # 匹配"X-X级"格式
        match = re.search(r'(\d+-\d+)级', wind_str)
        if match:
            return match.group(1)

        # 匹配"X级"格式
        match = re.search(r'(\d+)级', wind_str)
        if match:
            return match.group(1)

        return "0"

    def parse_weather_data(self, html_content):
        """解析HTML内容中的天气数据"""
        # 创建BeautifulSoup对象
        soup = BeautifulSoup(html_content, 'html.parser')

        # 查找天气表格
        weather_table = soup.find('table', class_='weather-table')
        if not weather_table:
            print("未找到天气表格")
            return []

        weather_data = []
        rows = weather_table.find_all('tr')[1:]  # 跳过表头行

        for row in rows:
            # 跳过空行或分隔行
            if not row.find_all('td') or row.find('hr'):
                continue

            columns = row.find_all('td')
            if len(columns) < 4:
                continue

            # 解析日期
            date_element = columns[0].find('a')
            date_text = date_element.get_text(strip=True) if date_element else columns[0].get_text(strip=True)
            formatted_date = self.parse_date(date_text)
            if not formatted_date:
                continue

            # 解析天气状况
            weather_text = columns[1].get_text(strip=True).replace('\xa0', ' ')
            weather_parts = weather_text.split('/')
            day_weather = weather_parts[0].strip() if len(weather_parts) > 0 else ''
            night_weather = weather_parts[1].strip() if len(weather_parts) > 1 else day_weather
            # 解析温度
            high_temp, low_temp = self.parse_temperature(columns[2].get_text(strip=True))
            # 解析风力
            wind_text = columns[3].get_text(strip=True).replace('\xa0', ' ')
            wind_parts = wind_text.split('/')
            day_wind = wind_parts[0].strip() if len(wind_parts) > 0 else ''
            night_wind = wind_parts[1].strip() if len(wind_parts) > 1 else day_wind
            # 提取风力等级
            day_wind_level = self.extract_wind_level(day_wind)
            night_wind_level = self.extract_wind_level(night_wind)
            # 添加到结果列表
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
                # 添加随机延迟，避免频繁请求被封IP
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)

                print(f"正在爬取 {year}年{month}月 数据...")
                headers = self.generate_random_headers()

                # 发送HTTP请求
                response = self.session.get(url, headers=headers, timeout=15)

                # 检查响应状态
                if response.status_code != 200:
                    print(f"请求失败，状态码: {response.status_code}, 尝试 {attempt + 1}/{self.retry_count}")
                    continue

                # 尝试多种编码解析响应内容
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

    def crawl_weather_data(self, start_year=2021, end_year=2023):
        """
        爬取指定年份范围内的所有天气数据
        返回: 包含所有天气数据的DataFrame
        """
        # 获取所有需要爬取的月份URL
        month_urls = self.get_month_urls(start_year, end_year)
        print(f"准备爬取 {len(month_urls)} 个月份的数据...")

        all_data = []
        success_count = 0
        fail_count = 0

        # 使用线程池并发爬取数据
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有爬取任务
            futures = []
            for year, month, url in month_urls:
                futures.append(executor.submit(self.fetch_month_data, year, month, url))

            # 获取并处理结果
            for future in futures:
                month_data = future.result()
                if month_data:
                    all_data.extend(month_data)
                    success_count += 1
                else:
                    fail_count += 1

        print(f"爬取完成: 成功 {success_count} 个月份, 失败 {fail_count} 个月份")

        # 数据处理和格式化
        if not all_data:
            print("未爬取到任何数据")
            return pd.DataFrame()

        # 转换为DataFrame并处理日期
        df = pd.DataFrame(all_data)
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])  # 移除无效日期
        df = df.sort_values('日期')  # 按日期排序
        df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')  # 格式化日期

        return df


if __name__ == "__main__":
    # 创建爬虫实例
    spider = WeatherSpider()

    # 设置爬取年份范围
    START_YEAR = 2022
    END_YEAR = 2024

    # 执行爬取
    print(f"开始爬取 {START_YEAR}-{END_YEAR} 年大连天气数据...")
    weather_data = spider.crawl_weather_data(START_YEAR, END_YEAR)

    # 保存结果
    if not weather_data.empty:
        csv_path = f"大连天气数据{START_YEAR}_{END_YEAR}.csv"
        weather_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"数据已成功保存到 {csv_path}")



