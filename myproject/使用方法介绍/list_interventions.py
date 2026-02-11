"""
快速查看 Covasim 中所有可用的干预措施
运行此脚本可以列出所有可用的干预措施及其简要说明
"""

import covasim as cv
import inspect

def list_all_interventions():
    """列出所有可用的干预措施"""
    
    print("=" * 80)
    print("Covasim 所有可用的干预措施（Interventions）")
    print("=" * 80)
    print()
    
    # 从 interventions 模块获取所有干预措施
    from covasim import interventions
    
    # 定义干预措施分类
    intervention_categories = {
        "基础类": ["Intervention"],
        "通用干预": ["dynamic_pars", "sequence"],
        "传播率干预": ["change_beta", "clip_edges"],
        "测试干预": ["test_prob", "test_num"],
        "接触者追踪": ["contact_tracing"],
        "疫苗干预": ["simple_vaccine", "vaccinate_prob", "vaccinate_num", "vaccinate", "BaseVaccination"],
    }
    
    # 获取所有以字母开头的公共属性（排除私有属性）
    all_items = [item for item in dir(interventions) 
                 if not item.startswith('_') and 
                 inspect.isclass(getattr(interventions, item, None))]
    
    # 按分类显示
    for category, items in intervention_categories.items():
        print(f"\n【{category}】")
        print("-" * 80)
        for item in items:
            if hasattr(interventions, item):
                obj = getattr(interventions, item)
                if inspect.isclass(obj):
                    # 获取文档字符串的第一行作为简要说明
                    doc = inspect.getdoc(obj)
                    if doc:
                        first_line = doc.split('\n')[0].strip()
                        print(f"  • {item:20s} - {first_line}")
                    else:
                        print(f"  • {item:20s} - (无文档)")
    
    # 显示其他未分类的干预措施
    other_items = [item for item in all_items 
                   if not any(item in items for items in intervention_categories.values())]
    
    if other_items:
        print(f"\n【其他干预措施】")
        print("-" * 80)
        for item in other_items:
            obj = getattr(interventions, item)
            if inspect.isclass(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    first_line = doc.split('\n')[0].strip()
                    print(f"  • {item:20s} - {first_line}")
                else:
                    print(f"  • {item:20s} - (无文档)")
    
    print("\n" + "=" * 80)
    print("查看详细文档的方法：")
    print("  1. help(cv.test_prob)  # 查看某个干预措施的详细文档")
    print("  2. 查看源代码: covasim/interventions.py")
    print("  3. 查看示例: examples/t05_*.py, examples/t08_*.py")
    print("=" * 80)


def show_intervention_help(intervention_name):
    """显示特定干预措施的详细帮助"""
    if hasattr(cv, intervention_name):
        obj = getattr(cv, intervention_name)
        print(f"\n{'=' * 80}")
        print(f"{intervention_name} 的详细文档")
        print(f"{'=' * 80}\n")
        help(obj)
    else:
        print(f"错误: 找不到干预措施 '{intervention_name}'")
        print(f"可用的干预措施: {[x for x in dir(cv) if not x.startswith('_') and 'test' in x.lower() or 'vacc' in x.lower() or 'beta' in x.lower() or 'trace' in x.lower()]}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了参数，显示特定干预措施的帮助
        intervention_name = sys.argv[1]
        show_intervention_help(intervention_name)
    else:
        # 否则列出所有干预措施
        list_all_interventions()
        
        print("\n\n提示: 运行 'python list_interventions.py <干预措施名称>' 查看详细文档")
        print("例如: python list_interventions.py test_prob")
