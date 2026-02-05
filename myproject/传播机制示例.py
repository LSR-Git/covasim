"""
Covasim 病毒传播机制示例
演示传播概率的计算过程
"""

import numpy as np
import covasim as cv

def demonstrate_transmission():
    """演示传播机制"""
    
    print("=" * 80)
    print("Covasim 病毒传播机制演示")
    print("=" * 80)
    print()
    
    # 创建简单的模拟
    sim = cv.Sim(pop_size=100, pop_infected=5, n_days=10, verbose=0)
    sim.initialize()
    
    # 获取第一天的人员状态
    people = sim.people
    t = 0
    
    print("【1. 初始状态】")
    print(f"  总人数: {len(people)}")
    print(f"  感染者数: {people.infectious.sum()}")
    print(f"  易感者数: {people.susceptible.sum()}")
    print()
    
    # 获取接触网络
    contacts = people.contacts
    print("【2. 接触网络】")
    for layer_key in contacts.keys():
        layer = contacts[layer_key]
        print(f"  层 '{layer_key}': {len(layer)} 条接触边")
    print()
    
    # 计算传播参数
    print("【3. 传播参数】")
    beta = sim['beta']
    print(f"  基础传播率 (beta): {beta}")
    
    # 获取第一层的参数
    first_layer_key = list(contacts.keys())[0]
    beta_layer = sim['beta_layer'][first_layer_key]
    print(f"  层传播权重 (beta_layer['{first_layer_key}']): {beta_layer}")
    print()
    
    # 演示传播概率计算
    print("【4. 传播概率计算示例】")
    print("  假设有一条接触边：感染者 A → 易感者 B")
    print()
    
    # 找到一对感染者和易感者
    inf_inds = np.where(people.infectious)[0]
    sus_inds = np.where(people.susceptible)[0]
    
    if len(inf_inds) > 0 and len(sus_inds) > 0:
        inf_person = inf_inds[0]
        sus_person = sus_inds[0]
        
        print(f"  感染者 A (ID={inf_person}):")
        print(f"    - 有症状: {people.symptomatic[inf_person]}")
        print(f"    - 隔离: {people.isolated[inf_person]}")
        print(f"    - 检疫: {people.quarantined[inf_person]}")
        print()
        
        print(f"  易感者 B (ID={sus_person}):")
        print(f"    - 易感: {people.susceptible[sus_person]}")
        print(f"    - 隔离: {people.isolated[sus_person]}")
        print(f"    - 检疫: {people.quarantined[sus_person]}")
        print()
        
        # 计算相对传播能力
        asymp_factor = sim['asymp_factor']
        iso_factor = sim['iso_factor'][first_layer_key]
        quar_factor = sim['quar_factor'][first_layer_key]
        
        # 感染者因素
        is_symp = people.symptomatic[inf_person]
        is_iso = people.isolated[inf_person]
        is_quar = people.quarantined[inf_person]
        
        f_asymp = 1.0 if is_symp else asymp_factor
        f_iso = iso_factor if is_iso else 1.0
        f_quar = quar_factor if is_quar else 1.0
        
        rel_trans = 1.0 * f_asymp * f_iso * f_quar * beta_layer
        
        print("  【传播能力计算】")
        print(f"    基础传播能力: 1.0")
        print(f"    症状因子 (f_asymp): {f_asymp:.2f} {'(有症状)' if is_symp else '(无症状)'}")
        print(f"    隔离因子 (f_iso): {f_iso:.2f} {'(隔离)' if is_iso else '(未隔离)'}")
        print(f"    检疫因子 (f_quar): {f_quar:.2f} {'(检疫)' if is_quar else '(未检疫)'}")
        print(f"    层权重 (beta_layer): {beta_layer:.2f}")
        print(f"    → 相对传播能力 (rel_trans): {rel_trans:.4f}")
        print()
        
        # 易感者因素
        is_sus = people.susceptible[sus_person]
        is_quar_sus = people.quarantined[sus_person]
        
        # 假设没有免疫力（简化）
        immunity = 0.0
        f_quar_sus = quar_factor if is_quar_sus else 1.0
        
        rel_sus = 1.0 * f_quar_sus * (1 - immunity)
        
        print("  【易感性计算】")
        print(f"    基础易感性: 1.0")
        print(f"    检疫因子 (f_quar): {f_quar_sus:.2f} {'(检疫)' if is_quar_sus else '(未检疫)'}")
        print(f"    免疫力: {immunity:.2f} (假设无免疫力)")
        print(f"    → 相对易感性 (rel_sus): {rel_sus:.4f}")
        print()
        
        # 最终传播概率
        layer_beta = 1.0  # 假设接触边权重为 1.0
        viral_load = 1.0  # 假设病毒载量为 1.0（简化）
        
        transmission_prob = beta * layer_beta * rel_trans * rel_sus * viral_load
        
        print("  【最终传播概率】")
        print(f"    P(传播) = β × layer_beta × rel_trans × rel_sus × viral_load")
        print(f"    P(传播) = {beta:.4f} × {layer_beta:.2f} × {rel_trans:.4f} × {rel_sus:.4f} × {viral_load:.2f}")
        print(f"    P(传播) = {transmission_prob:.6f} ({transmission_prob*100:.4f}%)")
        print()
        
        # 模拟传播
        print("  【传播判断】")
        random_value = np.random.random()
        transmission_occurred = random_value < transmission_prob
        print(f"    随机数: {random_value:.6f}")
        print(f"    传播阈值: {transmission_prob:.6f}")
        print(f"    结果: {'✓ 发生传播' if transmission_occurred else '✗ 未发生传播'}")
        print()
    
    print("=" * 80)
    print("【关键要点】")
    print("1. 传播是概率性的，不是确定性的")
    print("2. 传播概率受多种因素影响：")
    print("   - 基础传播率 (beta)")
    print("   - 接触层类型 (beta_layer)")
    print("   - 感染者状态（症状、隔离、检疫）")
    print("   - 易感者状态（易感性、隔离、免疫力）")
    print("   - 病毒载量（随时间变化）")
    print("3. 每条接触边独立判断是否传播")
    print("4. 传播只发生在有接触关系的两个人之间")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_transmission()
