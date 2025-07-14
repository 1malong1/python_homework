import requests
import pandas as pd
from time import sleep
import random

# 创建数据存储列表
数据 = []
#DaTa
# 列名映射（更新出生地相关字段）Column_nmap
列名映射 = {
    'hs_Rank_Rich_Ranking': '排名',
    'hs_Rank_Rich_Ranking_Change': '排名变化',
    'hs_Rank_Rich_ChaName_Cn': '中文全名',
    'hs_Rank_Rich_ChaName_En': '英文全名',
    'hs_Character.hs_Character_Age': '年龄',
    # 更新出生地字段路径
    'hs_Character.hs_Character_BirthPlace_Cn': '出生地中文',
    'hs_Character.hs_Character_BirthPlace_En': '出生地英文',
    'hs_Character.hs_Character_Gender': '性别',
    'hs_Character.hs_Character_Birthday': '出生日期',
    'hs_Character.hs_Character_Education_Cn': '教育程度中文',
    'hs_Character.hs_Character_Education_En': '教育程度英文',
    'hs_Character.hs_Character_School_Cn': '学校中文',
    'hs_Character.hs_Character_School_En': '学校英文',
    'hs_Character.hs_Character_Permanent_Cn': '常居地中文',
    'hs_Character.hs_Character_Permanent_En': '常居地英文',
    'hs_Rank_Rich_Photo': '照片',
    'hs_Rank_Rich_ComName_Cn': '公司中文全名',
    'hs_Rank_Rich_ComName_En': '公司英文全名',
    'hs_Rank_Rich_ComHeadquarters_Cn': '公司总部地中文',
    'hs_Rank_Rich_ComHeadquarters_En': '公司总部地英文',
    'hs_Rank_Rich_Industry_Cn': '所在行业中文',
    'hs_Rank_Rich_Industry_En': '所在行业英文',
    'hs_Rank_Rich_Relations': '组织结构',
    'hs_Rank_Rich_Wealth': '人民币财富值',
    'hs_Rank_Rich_Wealth_Change': '财富值变化',
    'hs_Rank_Rich_Wealth_USD': '美元财富值',
    'hs_Rank_Rich_Year': '年份'
}

# 处理嵌套结构的函数
def get_nested_value(data, path):
    """从嵌套结构中获取值，处理列表和字典混合的情况"""
    parts = path.split('.')
    value = data
    for part in parts:
        if isinstance(value, list):
            # 如果是列表，尝试获取第一个元素
            if len(value) > 0:
                value = value[0]
            else:
                return None
        if isinstance(value, dict):
            value = value.get(part)
            if value is None:
                return None
        else:
            # 如果既不是列表也不是字典，无法继续深入
            return value
    return value

# 循环请求1~15页
for page in range(1, 16):
    sleep_seconds = random.uniform(1, 2)
    print(f'开始等待{sleep_seconds:.2f}秒')
    sleep(sleep_seconds)
    print(f'开始爬取第{page}页')

    offset = (page - 1) * 200
    url = f'https://www.hurun.net/zh-CN/Rank/HsRankDetailsList?num=ODBYW2BI&search=&offset={offset}&limit=200'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept_language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'content_type': 'application/json',
        'referer': 'https://www.hurun.net/zh-CN/Rank/HsRankDetails?pagetype=rich'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_data = response.json()

        if 'rows' in json_data and isinstance(json_data['rows'], list):
            print(f"第{page}页包含{len(json_data['rows'])}条数据")
            for item in json_data['rows']:
                # 转换单个数据项
                transformed_item = {}
                for site_field, 中文列名 in 列名映射.items():
                    # 使用改进的嵌套值获取函数
                    transformed_item[中文列名] = get_nested_value(item, site_field)

                # 处理特殊字段
                if transformed_item.get('性别') == '先生':
                    transformed_item['性别'] = '男'
                elif transformed_item.get('性别') == '女士':
                    transformed_item['性别'] = '女'

                数据.append(transformed_item)
        else:
            print(f"第{page}页数据格式异常或为空")
            print(f"数据结构: {json_data.keys()}")

        print(f"第{page}页数据处理完成")
    except Exception as e:
        print(f"处理第{page}页数据时出错: {e}")

# 保存数据到CSV
if 数据:
    df = pd.DataFrame(数据)

    # 确保所有需要的列都存在Column_cn
    for 中文列名 in 列名映射.values():
        if 中文列名 not in df.columns:
            df[中文列名] = None

    # 调整列顺序
    df = df[list(列名映射.values())]

    try:
        df.to_csv('2024胡润富豪榜.csv', index=False, encoding='utf_8_sig')
        print(f"数据已成功保存到 '2024胡润富豪榜.csv'，共{len(df)}条记录")

        # 打印前几行预览
        print("\n数据前几行预览:")
        print(df.head().to_csv(sep='\t', na_rep='nan'))
    except Exception as e:
        print(f"保存数据时出错: {e}")
else:
    print("未获取到任何数据，无法生成CSV文件")

