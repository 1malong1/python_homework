import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# 设置全局字体，确保中文能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 针对Windows系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300


class WeatherVisualizer:
    """天气数据可视化类，用于分析和可视化天气数据"""

    def __init__(self, data_path):
        """初始化可视化器并加载数据"""
        self.data = self._load_data(data_path)
        if not self.data.empty:
            self._preprocess_data()

    def _load_data(self, data_path):
        """加载天气数据"""
        try:
            if not os.path.exists(data_path):
                print(f"错误: 文件 {data_path} 不存在")
                return pd.DataFrame()

            # 尝试读取CSV文件
            data = pd.read_csv(data_path)
            print(f"成功加载数据，共 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            return pd.DataFrame()

    def _preprocess_data(self):
        """预处理数据"""
        # 转换日期列
        self.data['日期'] = pd.to_datetime(self.data['日期'])

        # 提取年月信息
        self.data['年份'] = self.data['日期'].dt.year
        self.data['月份'] = self.data['日期'].dt.month

        # 创建年月组合列，用于分组
        self.data['年月'] = self.data['日期'].dt.strftime('%Y-%m')

        # 确保温度和风力列是数值类型
        self.data['最高温度'] = pd.to_numeric(self.data['最高温度'], errors='coerce')
        self.data['最低温度'] = pd.to_numeric(self.data['最低温度'], errors='coerce')

        # 合并白天和夜晚天气状况，用逗号分隔
        self.data['天气状况'] = self.data.apply(
            lambda row: f"{row['白天天气']},{row['夜晚天气']}", axis=1
        )

    def plot_monthly_temperature(self, output_path=None):
        if self.data.empty:
            print("没有数据可用于绘图")
            return

        monthly_temp = self.data.groupby('月份').agg({
            '最高温度': 'mean',
            '最低温度': 'mean'
        }).reset_index()

        plt.figure(figsize=(12, 6))
        sns.lineplot(x='月份', y='最高温度', data=monthly_temp, marker='o', label='平均最高温度')
        sns.lineplot(x='月份', y='最低温度', data=monthly_temp, marker='o', label='平均最低温度')

        # 设置图表标题和轴标签
        plt.title('近三年月平均气温变化趋势')
        plt.xlabel('月份')
        plt.ylabel('温度 (°C)')

        # 设置x轴刻度为1-12月
        plt.xticks(range(1, 13))

        # 添加图例和网格线
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 优化布局
        plt.tight_layout()

        # 保存图表或显示
        if output_path:
            plt.savefig(output_path)
            print(f"月平均气温变化图已保存至 {output_path}")
        else:
            plt.show()

    def plot_wind_distribution(self, output_path=None):
        """绘制近三年风力情况分布图"""
        if self.data.empty:
            print("没有数据可用于绘图")
            return

        # 合并白天和夜晚风力
        all_wind_levels = pd.concat([
            self.data['白天风力'].dropna(),
            self.data['夜晚风力'].dropna()
        ])

        # 统计每个风力等级出现的次数
        wind_counts = all_wind_levels.value_counts().reset_index()
        wind_counts.columns = ['风力等级', '天数']

        # 过滤掉无效的风力等级
        valid_wind = wind_counts[wind_counts['风力等级'] != '0']

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 绘制风力分布柱状图
        sns.barplot(x='天数', y='风力等级', data=valid_wind, orient='h')

        # 添加数据标签
        for i, v in enumerate(valid_wind['天数']):
            plt.text(v + 1, i, str(v), va='center')

        # 设置图表标题和轴标签
        plt.title('近三年风力情况分布')
        plt.xlabel('天数')
        plt.ylabel('风力等级')

        # 优化布局
        plt.tight_layout()

        # 保存图表或显示
        if output_path:
            plt.savefig(output_path)
            print(f"风力情况分布图已保存至 {output_path}")
        else:
            plt.show()

    def plot_weather_condition_distribution(self, output_path=None):
        if self.data.empty:
            print("没有数据可用于绘图")
            return

        # 定义主要天气类型及其包含的天气状况
        weather_mapping = {
            '晴天': ['晴', '晴朗'],
            '多云': ['多云', '少云', '晴间多云'],
            '阴天': ['阴', '阴天'],
            '雨天': ['雨', '小雨', '中雨', '大雨', '暴雨', '雷阵雨', '阵雨'],
            '雪天': ['雪', '小雪', '中雪', '大雪', '暴雪', '雨夹雪'],
            '雾/霾': ['雾', '薄雾', '霾', '雾霾', '扬沙', '浮尘']
        }

        # 统计所有天气状况出现的次数
        all_weather_conditions = []

        # 处理白天天气
        for condition in self.data['白天天气'].dropna():
            all_weather_conditions.append(condition.strip())

        # 处理夜晚天气
        for condition in self.data['夜晚天气'].dropna():
            all_weather_conditions.append(condition.strip())

        # 统计每种天气状况出现的次数
        condition_counts = pd.Series(all_weather_conditions).value_counts().reset_index()
        condition_counts.columns = ['天气状况', '天数']

        # 将天气状况归类到主要天气类型
        categorized_weather = defaultdict(int)

        for _, row in condition_counts.iterrows():
            condition = row['天气状况']
            count = row['天数']
            categorized = False

            # 检查是否属于已定义的天气类型
            for main_type, subtypes in weather_mapping.items():
                if any(sub in condition for sub in subtypes):
                    categorized_weather[main_type] += count
                    categorized = True
                    break

            # 如果不属于任何已定义类型，则归为"其他"
            if not categorized:
                categorized_weather['其他'] += count

        # 转换为DataFrame
        weather_df = pd.DataFrame(list(categorized_weather.items()), columns=['天气类型', '天数'])

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 绘制天气状况分布柱状图
        sns.barplot(x='天数', y='天气类型', data=weather_df, orient='h')

        # 添加数据标签
        for i, v in enumerate(weather_df['天数']):
            plt.text(v + 1, i, str(v), va='center')

        # 设置图表标题和轴标签
        plt.title('近三年天气状况分布')
        plt.xlabel('天数')
        plt.ylabel('天气类型')

        # 优化布局
        plt.tight_layout()

        # 保存图表或显示
        if output_path:
            plt.savefig(output_path)
            print(f"天气状况分布图已保存至 {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # 设置数据文件路径
    DATA_PATH = "大连天气数据2022_2024.csv"  # 请确保此文件存在

    # 创建可视化器实例
    visualizer = WeatherVisualizer(DATA_PATH)

    # 绘制并保存图表
    if not visualizer.data.empty:
        # 月平均气温变化图
        visualizer.plot_monthly_temperature("月平均气温变化图.png")

        # 风力情况分布图
        visualizer.plot_wind_distribution("风力情况分布图.png")

        # 天气状况分布图
        visualizer.plot_weather_condition_distribution("天气状况分布图.png")