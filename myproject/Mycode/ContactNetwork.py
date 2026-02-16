import numpy as np
import covasim as cv
import Enums
import sciris as sc
import os
import matplotlib.pyplot as plt
import networkx as nx

def validate_countries_config(countries_config):
    '''
    校验国家配置字典，支持两种写法：

    1) 比例式（推荐）：值为正整数之比，自动归一化为概率
       例如：{'A': 2, 'B': 1} 表示 A:B = 2:1，即 A 占 2/3，B 占 1/3

    2) 概率式：各值之和必须为 1.0
       例如：{'A': 0.6, 'B': 0.4} 表示 A 占 60%，B 占 40%

    Args:
        countries_config: 国家配置字典，格式为：
            {
                'country_name': proportion_or_ratio,  # 比例式如 2、1 或概率如 0.6、0.4
                ...
            }

    Returns:
        tuple: (country_names, proportions) - 国家名列表和归一化后的比例列表（和为 1.0）

    Raises:
        TypeError: 如果 countries_config 不是字典类型，或比例不是数值类型
        ValueError: 如果 countries_config 为空，或比例为负数，或概率式总和不等于 1.0
    '''
    # 校验是否为字典类型
    if not isinstance(countries_config, dict):
        raise TypeError(f"countries_config 必须是字典类型，当前类型: {type(countries_config)}")
    
    # 校验是否为空 如果没有指定国家，则默认指定一个国家
    if len(countries_config) == 0:
        countries_config = {
            'A': 1.0
        }
    
    # 提取国家名和比例
    country_names = list(countries_config.keys())
    proportions = np.array([float(v) for v in countries_config.values()])
    
    # 校验比例是否为数值类型且非负
    for country, prop in countries_config.items():
        if not isinstance(prop, (int, float)):
            raise TypeError(f"国家 '{country}' 的比例必须是数值类型，当前类型: {type(prop)}")
        if prop < 0:
            raise ValueError(f"国家 '{country}' 的比例不能为负数: {prop}")
    
    total = float(np.sum(proportions))
    if total <= 0:
        raise ValueError(
            f"countries_config 中所有比例之和必须大于 0，当前总和: {total}\n"
            f"当前配置: {countries_config}"
        )
    
    # 若总和不为 1.0，视为比例式（如 2:1），归一化为概率
    if abs(total - 1.0) > 1e-6:
        proportions = proportions / total
    proportions = proportions.tolist()
    
    return country_names, proportions

def create_custom_population(pop_size, layer_config, countries_config, seed=None):
    '''
    创建完全自定义的人口
    
    Args:
        pop_size: 人口大小
        layer_config: 层配置字典，格式为：
            {
                'layer_name': {
                    'n_contacts': 平均接触数,
                    'beta': 传播率,
                    'age_range': (min_age, max_age) 或 None 表示所有年龄,
                    'cluster_size': 如果是聚类结构，指定聚类大小；否则为 None
                }
            }
        countries_config: 国家配置字典。支持比例式如 {'A': 2, 'B': 1}（A:B=2:1）
            或概率式如 {'A': 0.6, 'B': 0.4}（总和须为 1.0），详见 validate_countries_config
        seed: 可选，人口与区内接触网生成的随机种子；为 None 时不固定
    '''
    if seed is not None:
        np.random.seed(seed)
    # 校验 countries_config 并获取国家名和比例列表
    country_names, proportions = validate_countries_config(countries_config)
    
    # 创建基本属性
    uids = np.arange(pop_size, dtype=cv.default_int)
    ages = np.random.uniform(18, 65, pop_size)
    sexes = np.random.binomial(1, 0.5, pop_size)
    
    # 根据 countries_config 生成 countries 数组
    # 使用 np.random.choice 根据比例随机分配
    countries = np.random.choice(country_names, size=pop_size, p=proportions)
    # 初始时 position = country，便于跨境时区分（流动者 position 可单独更新）
    positions = np.array(countries, dtype=object)

    # 创建接触网络
    contacts = cv.Contacts()
    layer_keys = []
    
    for layer_name, config in layer_config.items():
        layer_keys.append(layer_name)
        
        # 按 country 分组，只允许相同 country 的人之间建立连接
        unique_countries = np.unique(countries)
        all_p1 = []
        all_p2 = []
        
        # 为每个 country 分别生成网络
        for country in unique_countries:
            # 筛选出该 country 的所有人员索引
            country_mask = countries == country
            country_indices = np.where(country_mask)[0]
            
            if len(country_indices) == 0:
                continue  # 跳过空组
            
            # 在该 country 组内，根据年龄范围进一步筛选（如果有）
            if config.get('age_range') is not None:
                min_age, max_age = config['age_range']
                age_mask = (ages[country_indices] >= min_age) & (ages[country_indices] < max_age)
                filtered_indices = country_indices[age_mask]
            else:
                filtered_indices = country_indices
            
            if len(filtered_indices) == 0:
                continue  # 跳过没有符合年龄条件的人员的组
            
            # 根据网络类型生成该 country 组的接触网络
            if config.get('network_type') == Enums.NetWorkType.scale_free.name:
                # 使用无标度网络
                m = config.get('m_connections', 2)
                # 为这个 country 组生成无标度网络，使用 filtered_indices 作为映射
                country_contacts = cv.make_scale_free_contacts(
                    len(filtered_indices), 
                    m_connections=m, 
                    mapping=filtered_indices
                )
                
            elif config.get('network_type') == Enums.NetWorkType.microstructured.name:
                # 使用聚类结构
                cluster_size = config.get('cluster_size', 3.0)
                # 为这个 country 组生成微结构化网络
                temp_contacts = cv.make_microstructured_contacts(
                    len(filtered_indices), 
                    cluster_size=cluster_size
                )
                # 映射回原始索引
                country_contacts = {
                    'p1': filtered_indices[temp_contacts['p1']],
                    'p2': filtered_indices[temp_contacts['p2']]
                }
                # 如果有 beta 属性，也保留
                if 'beta' in temp_contacts:
                    country_contacts['beta'] = temp_contacts['beta']
                    
            elif config.get('network_type') == Enums.NetWorkType.random.name:
                # 使用随机接触
                n_contacts = config.get('n_contacts', 10)
                # 为这个 country 组生成随机网络
                country_contacts = cv.make_random_contacts(
                    len(filtered_indices), 
                    n=n_contacts, 
                    mapping=filtered_indices
                )
            else:
                # 未知的网络类型，跳过
                continue
            
            # 收集该 country 组的连接
            all_p1.extend(country_contacts['p1'])
            all_p2.extend(country_contacts['p2'])
        
        # 合并所有 country 组的连接
        if len(all_p1) > 0:
            layer_contacts = {
                'p1': np.array(all_p1, dtype=cv.default_int),
                'p2': np.array(all_p2, dtype=cv.default_int)
            }
            # 如果有 beta 属性，也合并（通常随机网络没有，无标度和微结构化可能有）
            # 这里简化处理，如果需要可以进一步优化
        else:
            # 如果没有连接，创建空的网络
            layer_contacts = {
                'p1': np.array([], dtype=cv.default_int),
                'p2': np.array([], dtype=cv.default_int)
            }
        
        # 创建层
        layer = cv.Layer(**layer_contacts, label=layer_name)
        contacts.add_layer(**{layer_name: layer})
    
    # 创建人口字典
    popdict = {
        
        # 基本属性
        'uid': uids,
        'age': ages,
        'sex': sexes,
        'contacts': contacts,
        'layer_keys': layer_keys,

        # 添加自定义属性（如果需要，可以在函数参数中添加更多自定义属性）
        'country': countries,
        'position': positions,  # 初始等于 country，跨境时可单独更新以区分所在地
    }
    
    return popdict, layer_keys
