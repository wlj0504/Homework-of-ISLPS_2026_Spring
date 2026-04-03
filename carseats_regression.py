# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ISLP import load_data


def main():
    print("=" * 60)
    print("加载 Carseats 数据集...")
    # 从 ISLP 库加载内置数据集
    carseats = load_data('Carseats')

    # ==========================================
    # (a) 拟合包含 Price, Urban, 和 US 的多元回归模型
    # ==========================================
    print("\n--- (a) 拟合初始多元回归模型 ---")
    # statsmodels 会自动将定性变量 Urban 和 US 转换为哑变量 (Dummy variables)
    # 默认将字典序靠前的 'No' 作为基准组，生成 'Urban[T.Yes]' 和 'US[T.Yes]'
    model_a = smf.ols('Sales ~ Price + Urban + US', data=carseats).fit()
    print(model_a.summary().tables[1])  # 只打印系数表以保持简洁

    # ==========================================
    # (b) & (c) 模型解释与方程形式
    # ==========================================
    print("\n--- (b) & (c) 系数解释与模型方程 ---")
    print("模型方程可以写为:")
    print("Sales = beta_0 + beta_1 * Price + beta_2 * I(Urban=Yes) + beta_3 * I(US=Yes)")
    print("\n基于模型 summary，系数解释如下：")
    print("1. Price: 价格每增加1美元，销量平均下降约 0.054 个单位 (即 54 个安全座椅)。")
    print(
        "2. Urban[T.Yes]: 假设其他条件相同，城市地区门店的销量比乡村地区平均少 0.022 个单位。但由于 p 值很大，这个差异在统计上不显著。")
    print("3. US[T.Yes]: 假设其他条件相同，美国境内门店的销量比境外平均高 1.200 个单位。")

    # ==========================================
    # (d) 我们可以拒绝哪些预测变量的零假设 (H0: beta_j = 0)？
    # ==========================================
    print("\n--- (d) 假设检验 ---")
    print("查看 p 值 (P>|t| 列)：")
    print("我们可以拒绝 Price (p ≈ 0.000) 和 US (p ≈ 0.000) 的零假设，因为它们高度显著。")
    print("我们无法拒绝 Urban (p = 0.936) 的零假设。")

    # ==========================================
    # (e) 拟合只包含显著变量的较小模型
    # ==========================================
    print("\n--- (e) 拟合精简模型 ---")
    # 移除不显著的 Urban
    model_e = smf.ols('Sales ~ Price + US', data=carseats).fit()
    print(model_e.summary().tables[1])

    # ==========================================
    # (f) 模型 (a) 和 (e) 的拟合优度对比
    # ==========================================
    print("\n--- (f) 模型拟合优度比较 ---")
    print(f"模型 (a) 的 R-squared: {model_a.rsquared:.4f}, Adjusted R-squared: {model_a.rsquared_adj:.4f}")
    print(f"模型 (e) 的 R-squared: {model_e.rsquared:.4f}, Adjusted R-squared: {model_e.rsquared_adj:.4f}")
    print(
        "结论：两个模型的 R-squared 非常接近（约为 23.9%）。精简模型 (e) 的调整后 R 平方甚至略微高一点点，说明移除 Urban 没有损失任何有用的解释力，精简模型更好。")

    # ==========================================
    # (g) 模型 (e) 的 95% 置信区间
    # ==========================================
    print("\n--- (g) 95% 置信区间 ---")
    conf_intervals = model_e.conf_int(alpha=0.05)
    conf_intervals.columns = ['2.5%', '97.5%']
    print(conf_intervals)

    # ==========================================
    # (h) 离群点 (Outliers) 和高杠杆点 (High Leverage)
    # ==========================================
    print("\n--- (h) 异常值与高杠杆点分析 ---")

    # 计算学生化残差 (Studentized residuals)
    outlier_test = model_e.outlier_test()
    student_resid = outlier_test['student_resid']

    # 找出绝对值大于 3 的学生化残差作为可能的离群点
    outliers = student_resid[abs(student_resid) > 3]
    if len(outliers) > 0:
        print(f"发现可能的离群点 (学生化残差绝对值 > 3):\n{outliers}")
    else:
        print("未发现明显的离群点 (所有学生化残差绝对值均 < 3)。")

    # 获取杠杆值 (Leverage)
    influence = model_e.get_influence()
    leverage = influence.hat_matrix_diag
    p = 2  # 预测变量数量 (Price, US)
    n = len(carseats)
    # 高杠杆的经验法则阈值：2*(p+1)/n 或 3*(p+1)/n
    threshold = 3 * (p + 1) / n
    high_leverage = leverage[leverage > threshold]
    print(f"计算高杠杆阈值: {threshold:.4f}")
    print(f"发现高杠杆点的数量: {len(high_leverage)}")

    # 绘制影响图 (Influence Plot) 直观展示
    print("\n正在生成影响图，请查看弹出的绘图窗口...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.influence_plot(model_e, ax=ax, criterion="cooks")
    plt.title("Influence Plot: Studentized Residuals vs Leverage")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
