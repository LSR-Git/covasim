import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置科研风格的绘图参数
def set_science_style():
    plt.style.use('seaborn-v0_8-whitegrid') 
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimHei', 'Arial'], # 使用黑体支持中文
        'axes.unicode_minus': False,
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'figure.figsize': (8, 5),
        'axes.grid': False, # 去除网格
        'lines.linewidth': 2.5,
    })

def nab_growth_decay(length, growth_time=21, decay_rate1=np.log(2)/50, decay_time1=150, decay_rate2=np.log(2)/250, decay_time2=365):
    '''
    生成NAb生长和衰减的动力学曲线（变化量）。
    '''
    
    def f1(t, growth_time):
        '''简单的线性增长'''
        return (1.0 / growth_time) * t

    def f2(t, decay_time1, decay_time2, decay_rate1, decay_rate2):
        '''复杂的指数衰减'''
        decayRate = np.full(len(t), fill_value=decay_rate1)
        
        # 晚期衰减率
        mask_late = t > decay_time2
        decayRate[mask_late] = decay_rate2
        
        # 中期过渡衰减率
        mask_mid = (t > decay_time1) & (t <= decay_time2)
        if np.any(mask_mid):
            slowing = (1.0 / (decay_time2 - decay_time1)) * (decay_rate1 - decay_rate2)
            indices = np.arange(np.sum(mask_mid)) 
            decayRate[mask_mid] = decay_rate1 - slowing * indices

        # 积分计算 titre
        titre = np.zeros(len(t))
        for i in range(1, len(t)):
            titre[i] = titre[i-1] + decayRate[i]
            
        return np.exp(-titre)

    # 构造时间轴
    calc_length = length + 1
    t1 = np.arange(growth_time)
    t2 = np.arange(calc_length - growth_time)
    
    y1 = f1(t1, growth_time)
    y2 = f2(t2, decay_time1, decay_time2, decay_rate1, decay_rate2)
    
    y = np.concatenate([y1, y2])
    nab_kin = np.diff(y)[0:length]
    
    return nab_kin

def plot_nab_decay_curve(days=730):
    '''
    绘制NAb衰减示意图（中文标注）
    '''
    set_science_style()
    
    # 获取数据
    nab_kin = nab_growth_decay(length=days)
    reconstructed_nab = np.concatenate(([0], np.cumsum(nab_kin)))
    t = np.arange(len(reconstructed_nab))
    
    fig, ax = plt.subplots()
    
    # 绘制曲线
    ax.plot(t, reconstructed_nab, color='black')
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 去除刻度数值，保留刻度线或者也去除刻度线
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 设置轴标签（位置调整以模仿坐标轴末端标签）
    # Y轴标签放在顶部
    ax.set_ylabel('NAb水平', loc='top', rotation=0, labelpad=-10, fontsize=14)
    # X轴标签放在右侧
    ax.set_xlabel('时间', loc='right', labelpad=-10, fontsize=14)
    
    # 标注峰值
    peak_idx = np.argmax(reconstructed_nab)
    peak_val = reconstructed_nab[peak_idx]
    peak_day = t[peak_idx]
    
    # 添加峰值文字，稍微上移
    ax.text(peak_day, peak_val + 0.05, '峰值', ha='center', fontsize=14)
    
    # 底部区域标注 (使用 xaxis_transform 在轴下方定位)
    trans = ax.get_xaxis_transform()
    y_text_pos = -0.05 # 轴下方距离
    
    # 感染/接种 (在起始位置附近)
    ax.text(peak_day, y_text_pos, '感染/接种', transform=trans, ha='center', va='top', fontsize=12)
    
    # 衰减期 (在峰值和终点中间)
    decay_mid_day = peak_day + (days - peak_day) / 3
    ax.text(decay_mid_day, y_text_pos, '衰减期', transform=trans, ha='center', va='top', fontsize=12)
    
    # 基线 (在末尾)
    ax.text(days - 50, y_text_pos, '基线', transform=trans, ha='center', va='top', fontsize=12)
    
    # 可选：画一条虚线表示基线水平(接近0)
    # ax.axhline(y=0, color='gray', linestyle='--', linewidth=1) 
    
    plt.tight_layout()
    output_file = 'nab_decay_plot.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    plot_nab_decay_curve()
