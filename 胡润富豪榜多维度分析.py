import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re  # 用于正则表达式提取数字

# 创建保存图表的目录
os.makedirs('charts', exist_ok=True)

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取爬取的数据
try:
    df = pd.read_csv('2024胡润富豪榜.csv')
    print(f"数据加载成功，共{len(df)}条记录")
except FileNotFoundError:
    print("未找到 '2024胡润富豪榜.csv' 文件，请确保文件已成功生成。")
    exit()

# ---------------------- 关键优化：数据清洗（尤其是年龄列） ----------------------
print("\n数据清洗中...")


# 1. 处理年龄列：提取数字，过滤非合理年龄（1-120岁）
def clean_age(age_str):
    if pd.isna(age_str):
        return None
    # 尝试将数据转为字符串，提取其中的数字
    age_str = str(age_str)
    # 用正则表达式提取所有数字
    numbers = re.findall(r'\d+', age_str)
    if not numbers:
        return None  # 没有数字则视为缺失
    # 取第一个数字作为年龄（处理多个数字的异常情况）
    age = int(numbers[0])
    # 过滤不合理的年龄（1-120岁之间）
    return age if 1 <= age <= 120 else None


# 应用清洗函数
df['年龄'] = df['年龄'].apply(clean_age)


# 2. 处理性别列：标准化性别值
def clean_gender(gender):
    if pd.isna(gender):
        return None
    gender = str(gender).strip()
    # 统一性别表述（只保留男/女）
    if gender in ['男', '男性', 'M', 'm']:
        return '男'
    elif gender in ['女', '女性', 'F', 'f']:
        return '女'
    else:
        return None  # 其他值视为缺失


df['性别'] = df['性别'].apply(clean_gender)

# 3. 处理出生地列：去除空值和异常值
df['出生地中文'] = df['出生地中文'].apply(lambda x: x if pd.notna(x) and str(x).strip() else None)

# 数据质量分析
print("\n数据清洗后质量分析：")
for col in ['年龄', '性别', '出生地中文']:
    missing = df[col].isnull().sum()
    print(f"- {col} 列缺失值: {missing} ({missing / len(df):.2%})")

# 年龄分布直方图
valid_ages = df['年龄'].dropna()
if not valid_ages.empty:
    average_age = valid_ages.mean()
    median_age = valid_ages.median()
    mode_age = valid_ages.mode()[0] if not valid_ages.mode().empty else None
    plt.figure(figsize=(18, 8))
    ax = sns.histplot(valid_ages, bins=range(20, 111, 5), kde=True, kde_kws={'bw_adjust': 0.5})
    plt.xlim(15, 115)
    ages = range(20, 111, 5)
    age_labels = [f"{i}-{i + 4}" for i in ages[:-1]] + ["110+"]
    plt.xticks(ages, age_labels, fontsize=10)
    plt.title('富豪年龄分布直方图', fontsize=16)
    plt.xlabel('年龄区间', fontsize=14)
    plt.ylabel('人数', fontsize=14)  # 恢复为人数更直观
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 标注统计量
    plt.axvline(x=average_age, color='green', linestyle='--',
                label=f'平均值: {average_age:.1f}岁')
    plt.axvline(x=median_age, color='orange', linestyle='--',
                label=f'中位数: {median_age:.1f}岁')
    if mode_age:
        plt.axvline(x=mode_age, color='red', linestyle='--',
                    label=f'众数: {mode_age}岁')
    # 添加具体数字标注
    for rect in ax.patches:
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.5,
                    int(height), ha='center', va='bottom', fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('charts/富豪-年龄图.png', dpi=300)
    plt.close()
    print("年龄分布图表生成成功")
else:
    print("警告：年龄数据全部缺失，无法生成年龄分布图表")

# 性别分布饼图
valid_genders = df['性别'].dropna()
if not valid_genders.empty:
    gender_counts = valid_genders.value_counts()
    plt.figure(figsize=(8, 8))
    # 饼图中显示百分比和实际人数
    plt.pie(gender_counts, labels=gender_counts.index,
            autopct=lambda p: f'{p:.1f}%\n({int(p * sum(gender_counts) / 100)})',
            textprops={'fontsize': 12}, pctdistance=0.8, startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('富豪性别分布饼图', fontsize=16)
    plt.tight_layout()
    plt.savefig('charts/富豪-性别图.png', dpi=300)
    plt.close()
    print("性别分布图表生成成功")
else:
    print("警告：性别数据全部缺失，无法生成性别分布图表")

# 出生地分布柱状图
valid_birthplaces = df['出生地中文'].dropna()
if not valid_birthplaces.empty:
    top_10_birthplaces = valid_birthplaces.value_counts().head(10)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=top_10_birthplaces.index, y=top_10_birthplaces.values,
                     palette='viridis')
    plt.title('富豪出生地TOP10分布', fontsize=16)
    plt.xlabel('出生地', fontsize=14)
    plt.ylabel('人数', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 添加具体数字标注
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.5,
                f"{int(height)}", ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, pad=2))
    plt.tight_layout()
    plt.savefig('charts/富豪-出生地图.png', dpi=300)
    plt.close()
    print("出生地分布图表生成成功")
else:
    print("警告：出生地数据全部缺失，无法生成出生地分布图表")

# ---------------------- 新增可视化图表 ----------------------

# 1. 行业分布TOP10柱状图
valid_industries = df['所在行业中文'].dropna()
if not valid_industries.empty:
    top_10_industries = valid_industries.value_counts().head(10)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=top_10_industries.values, y=top_10_industries.index,
                     palette='magma')
    plt.title('富豪所在行业TOP10分布', fontsize=16)
    plt.xlabel('富豪数量', fontsize=14)
    plt.ylabel('行业', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    # 添加具体数字标注
    for i, v in enumerate(top_10_industries.values):
        ax.text(v + 0.5, i, f"{v}", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('charts/富豪-行业分布TOP10.png', dpi=300)
    plt.close()
    print("行业分布TOP10图表生成成功")
else:
    print("警告：行业数据全部缺失，无法生成行业分布图表")

# 2. 财富值分布直方图（取前100名避免分布过于分散）
df['人民币财富值'] = pd.to_numeric(df['人民币财富值'], errors='coerce')
valid_wealth = df['人民币财富值'].dropna()
if not valid_wealth.empty:
    top_100_wealth = valid_wealth.nlargest(100)  # 取财富值前100名
    plt.figure(figsize=(16, 8))
    ax = sns.histplot(top_100_wealth, bins=20, kde=True, color='green')
    plt.title('前100名富豪财富值分布（人民币）', fontsize=16)
    plt.xlabel('财富值（亿元）', fontsize=14)
    plt.ylabel('人数', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 标注统计量
    avg_wealth = top_100_wealth.mean()
    plt.axvline(x=avg_wealth, color='red', linestyle='--',
                label=f'平均财富: {avg_wealth:.1f}亿元')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('charts/富豪-财富值分布.png', dpi=300)
    plt.close()
    print("财富值分布图表生成成功")
else:
    print("警告：财富值数据全部缺失，无法生成财富值分布图表")

# 3. 财富变化TOP10与BOTTOM10对比图
df['财富值变化'] = pd.to_numeric(df['财富值变化'].str.replace('%', ''), errors='coerce')
valid_change = df.dropna(subset=['财富值变化', '中文全名'])
if len(valid_change) >= 10:
    # 取财富增长最快和下降最快的各10名
    top_growth = valid_change.nlargest(10, '财富值变化')[['中文全名', '财富值变化']]
    top_decline = valid_change.nsmallest(10, '财富值变化')[['中文全名', '财富值变化']]

    plt.figure(figsize=(18, 10))

    # 上半部分：增长最快
    plt.subplot(2, 1, 1)
    ax1 = sns.barplot(x='财富值变化', y='中文全名', data=top_growth, palette='rocket')
    plt.title('财富增长最快的10位富豪（%）', fontsize=14)
    plt.xlabel('财富增长率（%）', fontsize=12)
    plt.ylabel('姓名', fontsize=12)
    for i, v in enumerate(top_growth['财富值变化']):
        ax1.text(v + 1, i, f"{v}%", va='center', fontsize=10)

    # # 下半部分：下降最快
    plt.subplot(2, 1, 2)
    ax2 = sns.barplot(x='财富值变化', y='中文全名', data=top_decline, palette='mako')
    plt.title('财富下降最快的10位富豪（%）', fontsize=14)
    plt.xlabel('财富变化率（%）', fontsize=12)
    plt.ylabel('姓名', fontsize=12)
    # 调整x轴范围，左侧预留标签空间
    x_min = top_decline['财富值变化'].min()
    plt.xlim(x_min * 1.3, 0)  # 左侧预留30%空间（因数值为负，范围是从更小的负数到0）
    for i, v in enumerate(top_decline['财富值变化']):
        # 文字左对齐，距离柱形左侧有一定偏移
        ax2.text(v * 1.0, i, f"{v}%", va='center', fontsize=10, ha='right')

    plt.tight_layout()
    plt.savefig('charts/富豪-财富变化TOP10对比.png', dpi=300)
    plt.close()
    print("财富变化对比图表生成成功")
else:
    print("警告：财富变化数据不足，无法生成财富变化对比图表")

valid_education = df['教育程度中文'].dropna()
if not valid_education.empty:
    education_counts = valid_education.value_counts()
    plt.figure(figsize=(10, 10))
    plt.pie(education_counts, labels=education_counts.index,
            autopct=lambda p: f'{p:.1f}%\n({int(p * sum(education_counts) / 100)})',
            textprops={'fontsize': 12}, pctdistance=0.85, startangle=140,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('富豪教育程度分布', fontsize=16)
    plt.tight_layout()
    plt.savefig('charts/富豪-教育程度分布.png', dpi=300)
    plt.close()
    print("教育程度分布图表生成成功")
else:
    print("警告：教育程度数据全部缺失，无法生成教育程度分布图表")

valid_residence = df['常居地中文'].dropna()
if not valid_residence.empty:
    top_10_residence = valid_residence.value_counts().head(10)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=top_10_residence.index, y=top_10_residence.values,
                     palette='cividis')
    plt.title('富豪常居地TOP10分布', fontsize=16)
    plt.xlabel('常居地', fontsize=14)
    plt.ylabel('人数', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.5,
                f"{int(height)}", ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, pad=2))
    plt.tight_layout()
    plt.savefig('charts/富豪-常居地TOP10.png', dpi=300)
    plt.close()
    print("常居地分布图表生成成功")
else:
    print("警告：常居地数据全部缺失，无法生成常居地分布图表")

print("\n所有图表已成功保存到 'charts' 文件夹")



