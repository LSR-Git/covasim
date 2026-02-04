'''
测试 countries_config 参数的功能和校验
'''
import numpy as np
import ContactNetwork
import Enums

# 测试配置
layer_config = {
    'test_layer': {
        'network_type': Enums.NetWorkType.random.name,
        'n_contacts': 5,
        'beta': 0.3,
    }
}

print("="*60)
print("测试1: 正常情况 - 比例总和等于1.0")
print("="*60)
try:
    countries_config = {'A': 0.6, 'B': 0.4}
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    countries = popdict['country']
    unique, counts = np.unique(countries, return_counts=True)
    print(f"✓ 成功创建人口")
    print(f"  配置: {countries_config}")
    for country, count in zip(unique, counts):
        actual_prop = count / 100
        expected_prop = countries_config[country]
        print(f"  {country}: {count}人 (期望: {expected_prop:.1%}, 实际: {actual_prop:.1%})")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n" + "="*60)
print("测试2: 三个国家 - 比例总和等于1.0")
print("="*60)
try:
    countries_config = {'A': 0.5, 'B': 0.3, 'C': 0.2}
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    countries = popdict['country']
    unique, counts = np.unique(countries, return_counts=True)
    print(f"✓ 成功创建人口")
    print(f"  配置: {countries_config}")
    for country, count in zip(unique, counts):
        actual_prop = count / 100
        expected_prop = countries_config[country]
        print(f"  {country}: {count}人 (期望: {expected_prop:.1%}, 实际: {actual_prop:.1%})")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n" + "="*60)
print("测试3: 错误情况 - 比例总和大于1.0")
print("="*60)
try:
    countries_config = {'A': 0.6, 'B': 0.5}  # 总和=1.1 > 1.0
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    print(f"✗ 应该报错但没有报错")
except ValueError as e:
    print(f"✓ 正确捕获错误: {e}")
except Exception as e:
    print(f"✗ 捕获了错误但类型不对: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("测试4: 错误情况 - 比例总和小于1.0")
print("="*60)
try:
    countries_config = {'A': 0.6, 'B': 0.3}  # 总和=0.9 < 1.0
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    print(f"✗ 应该报错但没有报错")
except ValueError as e:
    print(f"✓ 正确捕获错误: {e}")
except Exception as e:
    print(f"✗ 捕获了错误但类型不对: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("测试5: 错误情况 - 空字典")
print("="*60)
try:
    countries_config = {}
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    print(f"✗ 应该报错但没有报错")
except ValueError as e:
    print(f"✓ 正确捕获错误: {e}")
except Exception as e:
    print(f"✗ 捕获了错误但类型不对: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("测试6: 错误情况 - 负数比例")
print("="*60)
try:
    countries_config = {'A': 0.6, 'B': -0.1, 'C': 0.5}
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    print(f"✗ 应该报错但没有报错")
except ValueError as e:
    print(f"✓ 正确捕获错误: {e}")
except Exception as e:
    print(f"✗ 捕获了错误但类型不对: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("测试7: 错误情况 - 非数值类型")
print("="*60)
try:
    countries_config = {'A': 0.6, 'B': '0.4'}  # B 是字符串
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    print(f"✗ 应该报错但没有报错")
except TypeError as e:
    print(f"✓ 正确捕获错误: {e}")
except Exception as e:
    print(f"✗ 捕获了错误但类型不对: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("测试8: 浮点数精度测试 - 接近1.0但不完全等于")
print("="*60)
try:
    # 使用浮点数，总和可能不完全等于1.0
    countries_config = {'A': 0.333333, 'B': 0.333333, 'C': 0.333334}  # 总和=1.0
    popdict, keys = ContactNetwork.create_custom_population(100, layer_config, countries_config)
    countries = popdict['country']
    unique, counts = np.unique(countries, return_counts=True)
    print(f"✓ 成功创建人口（使用了容差处理浮点数精度）")
    print(f"  配置: {countries_config}")
    for country, count in zip(unique, counts):
        actual_prop = count / 100
        expected_prop = countries_config[country]
        print(f"  {country}: {count}人 (期望: {expected_prop:.6f}, 实际: {actual_prop:.6f})")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n" + "="*60)
print("所有测试完成！")
print("="*60)
