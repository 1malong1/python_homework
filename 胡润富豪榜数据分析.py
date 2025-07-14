import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm


# 设置中文字体
def set_chinese_font():
    """设置中文字体为黑体，确保图表中的中文正常显示"""
    try:
        # 尝试查找系统中可用的黑体
        chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei']
        for font in chinese_fonts:
            if font in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams["font.family"] = font
                print(f"已设置中文字体: {font}")
                return True

        # 如果没有找到黑体，尝试使用系统默认
        plt.rcParams["font.family"] = ["sans-serif"]
        print("未找到中文字体，将使用系统默认字体")
        return False
    except Exception as e:
        print(f"设置字体时出错: {e}")
        return False


# 设置中文字体
set_chinese_font()
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取数据
try:
    df = pd.read_csv('2024胡润富豪榜.csv')
    print(f"成功加载数据，共{len(df)}条记录")
    print("\n数据基本信息：")
    df.info()

    # 检查必要的列是否存在
    required_columns = ['美元财富值', '年龄', '性别', '排名', '所在行业中文']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"错误：CSV 文件缺少必要的列：{', '.join(missing_columns)}")
        exit()

except FileNotFoundError:
    print("未找到CSV文件，请确保'2024胡润富豪榜.csv'文件在当前目录下")
    exit()

# 数据预处理
print("\n开始数据预处理...")


# 处理财富值列
def convert_wealth(value):
    """将财富值转换为数值类型，处理各种可能的格式"""
    if pd.isna(value):
        return 0.0

    value_str = str(value).strip()

    # 处理包含"亿美元"的情况
    if '亿美元' in value_str:
        value_str = value_str.replace('亿美元', '')

    # 处理包含其他货币符号的情况（如果有）
    value_str = value_str.replace('亿元', '').replace('$', '').replace(',', '')

    try:
        return float(value_str)
    except ValueError:
        print(f"无法转换财富值: {value}，将其设为0")
        return 0.0


# 应用转换函数
df['美元财富值'] = df['美元财富值'].apply(convert_wealth)
print(f"'美元财富值'列转换后数据类型: {df['美元财富值'].dtype}")

# 按行业统计富豪数量和总财富值
industry_stats = df.groupby('所在行业中文').agg(
    富豪数量=('排名', 'count'),
    总财富值=('美元财富值', 'sum'),
    平均财富值=('美元财富值', 'mean')
).reset_index()

# 按总财富值排序（取前15名）
top_industries = industry_stats.sort_values('总财富值', ascending=False).head(15)

# 按富豪数量排序（取前15名）
top_industries_by_count = industry_stats.sort_values('富豪数量', ascending=False).head(15)

# 计算各行业富豪数量占比
total_tycoons = df['排名'].count()
top_industries['富豪数量占比'] = (top_industries['富豪数量'] / total_tycoons * 100).map('{:.2f}%'.format)

# 计算各行业财富值占比
total_wealth = industry_stats['总财富值'].sum()
top_industries['财富值占比'] = (top_industries['总财富值'] / total_wealth * 100).map('{:.2f}%'.format)

# 修正：计算所有行业的平均财富值排名
industry_stats['平均财富值排名'] = industry_stats['平均财富值'].rank(ascending=False)

# 将所有行业的平均财富值排名合并到top_industries中
top_industries = pd.merge(
    top_industries,
    industry_stats[['所在行业中文', '平均财富值排名']],
    on='所在行业中文',
    how='left'
)

# 输出统计结果
print("\n各行业富豪数量和财富值统计（按总财富值排序，前15名）：")
print(top_industries[
          ['所在行业中文', '富豪数量', '富豪数量占比', '总财富值', '财富值占比', '平均财富值', '平均财富值排名']])

# 创建图表目录（如果不存在）
if not os.path.exists('charts'):
    os.makedirs('charts')

# 图表1：各行业总财富值分布（黑色柱状图 + 红色折线图）
plt.figure(figsize=(12, 8), dpi=300)  # 调整图表尺寸
ax1 = sns.barplot(x='所在行业中文', y='总财富值', data=top_industries, color='black')
plt.title('各行业总财富值分布（前15名）', fontsize=16)  # 标题字体大小16
plt.xlabel('行业', fontsize=14)  # X轴标签字体大小14
plt.ylabel('总财富值（亿美元）', fontsize=14)  # Y轴标签字体大小14
plt.xticks(rotation=45, ha='right', fontsize=12)  # X轴刻度字体大小12
plt.yticks(fontsize=12)  # Y轴刻度字体大小12

# 添加数据标签（居中显示，字体大小12）
for i, v in enumerate(top_industries['总财富值']):
    ax1.text(i, v + 10, f'{v:.1f}亿', ha='center', va='bottom', fontsize=12)

# 添加红色折线图
ax2 = ax1.twinx()
ax2.plot(top_industries['所在行业中文'], top_industries['富豪数量'], 'r-o', linewidth=1.5, markersize=5)
ax2.set_ylabel('富豪数量', color='red', fontsize=14)  # 第二条Y轴标签字体大小14
ax2.tick_params(axis='y', labelcolor='red', labelsize=12)  # 第二条Y轴刻度字体大小12
ax2.grid(False)  # 隐藏第二条Y轴的网格线

plt.tight_layout()
plt.savefig('charts/行业总财富值分布.png', dpi=300, bbox_inches='tight')
print("\n行业总财富值分布图表已保存为'charts/行业总财富值分布.png'")

# 图表2：各行业富豪数量分布（黑色柱状图 + 红色折线图）
plt.figure(figsize=(12, 8), dpi=300)  # 调整图表尺寸
ax3 = sns.barplot(x='所在行业中文', y='富豪数量', data=top_industries_by_count, color='black')
plt.title('各行业富豪数量分布（前15名）', fontsize=16)  # 标题字体大小16
plt.xlabel('行业', fontsize=14)  # X轴标签字体大小14
plt.ylabel('富豪数量', fontsize=14)  # Y轴标签字体大小14
plt.xticks(rotation=45, ha='right', fontsize=12)  # X轴刻度字体大小12
plt.yticks(fontsize=12)  # Y轴刻度字体大小12

# 添加数据标签（居中显示，字体大小12）
for i, v in enumerate(top_industries_by_count['富豪数量']):
    ax3.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)

# 添加红色折线图（平均财富值）
ax4 = ax3.twinx()
ax4.plot(top_industries_by_count['所在行业中文'], top_industries_by_count['平均财富值'], 'r-o', linewidth=1.5,
         markersize=5)
ax4.set_ylabel('平均财富值（亿美元）', color='red', fontsize=14)  # 第二条Y轴标签字体大小14
ax4.tick_params(axis='y', labelcolor='red', labelsize=12)  # 第二条Y轴刻度字体大小12
ax4.grid(False)  # 隐藏第二条Y轴的网格线

plt.tight_layout()
plt.savefig('charts/行业富豪数量分布.png', dpi=300, bbox_inches='tight')
print("\n行业富豪数量分布图表已保存为'charts/行业富豪数量分布.png'")

# 分析行业发展态势（修正后）
print("\n行业发展态势分析：")
print("1. 财富值最高的行业：", top_industries.iloc[0]['所在行业中文'],
      f"({top_industries.iloc[0]['总财富值']:.2f}亿美元)")
print("2. 富豪数量最多的行业：", top_industries_by_count.iloc[0]['所在行业中文'],
      f"({top_industries_by_count.iloc[0]['富豪数量']}人)")
print("3. 平均财富值最高的行业：", industry_stats.sort_values('平均财富值', ascending=False).iloc[0]['所在行业中文'],
      f"({industry_stats.sort_values('平均财富值', ascending=False).iloc[0]['平均财富值']:.2f}亿美元)")

# 保存统计结果到CSV
top_industries.to_csv('胡润富豪榜行业统计.csv', index=False, encoding='utf_8_sig')
print("\n详细统计结果已保存为'胡润富豪榜行业统计.csv'")

# 显示图表
plt.show()


