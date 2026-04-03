# -*- coding: utf-8 -*-
"""
Boston Housing 数据集习题 (a)~(i)
需要先安装:
    pip install ISLP
"""

import matplotlib.pyplot as plt
import pandas as pd
from ISLP import load_data
from pandas.plotting import scatter_matrix


def main():
    # =========================
    # (a) 载入 Boston 数据集
    # =========================
    Boston = load_data("Boston")

    print("=" * 60)
    print("(a) 成功载入 Boston 数据集")
    print("=" * 60)

    # =========================
    # (b) 行数、列数、含义
    # =========================
    n_rows, n_cols = Boston.shape

    print("\n" + "=" * 60)
    print("(b) 数据集的行数、列数")
    print("=" * 60)
    print(f"行数（样本数）: {n_rows}")
    print(f"列数（变量数）: {n_cols}")
    print("\n每一行表示一个波士顿地区的城镇/郊区（或普查区）的观测值。")
    print("每一列表示一个变量。")
    print("\n各列名称如下：")
    print(list(Boston.columns))

    # =========================
    # (c) 预测变量的成对散点图
    # =========================
    predictors = Boston.drop(columns=["medv"])

    print("\n" + "=" * 60)
    print("(c) 绘制预测变量的成对散点图")
    print("=" * 60)
    print("正在绘制散点图矩阵，请稍候...")

    # 由于全部变量一起画会非常密，这里挑选部分最常分析的变量
    selected_cols = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "lstat"]
    scatter_matrix(
        Boston[selected_cols],
        figsize=(16, 16),
        diagonal="hist",
        alpha=0.6
    )
    plt.suptitle("Boston 数据集部分预测变量成对散点图", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig("boston_scatter_matrix.png", dpi=300)
    plt.show()

    print("散点图已保存为: boston_scatter_matrix.png")
    print("\n(c) 可从图中观察到的大致现象：")
    print("1. rad 与 tax 往往呈明显正相关。")
    print("2. indus 与 nox 往往呈正相关。")
    print("3. rm 与 lstat 往往呈负相关。")
    print("4. dis 与 nox、indus 通常有一定负相关趋势。")
    print("5. crim、tax、rad 等变量分布右偏，存在较大的极端值。")

    # =========================
    # (d) 哪些预测变量与 crim 相关
    # =========================
    print("\n" + "=" * 60)
    print("(d) 与人均犯罪率 crim 的相关关系")
    print("=" * 60)

    corr_with_crim = Boston.corr(numeric_only=True)["crim"].sort_values(ascending=False)
    print("各数值变量与 crim 的相关系数：")
    print(corr_with_crim)

    print("\n说明：")
    print("相关系数为正，表示变量越大，犯罪率往往越高；")
    print("相关系数为负，表示变量越大，犯罪率往往越低。")
    print("通常可以看到 rad、tax、lstat、indus、nox 等与 crim 正相关较明显，")
    print("而 rm、dis、zn 等与 crim 往往负相关。")

    # =========================
    # (e) 是否有特别高的犯罪率、税率、师生比？并评论范围
    # =========================
    print("\n" + "=" * 60)
    print("(e) 高犯罪率 / 高税率 / 高师生比，以及各预测变量范围")
    print("=" * 60)

    # 各预测变量范围
    predictor_ranges = predictors.agg(["min", "max"]).T
    print("各预测变量的最小值和最大值：")
    print(predictor_ranges)

    # 最大值所在行
    max_crim_idx = Boston["crim"].idxmax()
    max_tax_idx = Boston["tax"].idxmax()
    max_ptratio_idx = Boston["ptratio"].idxmax()

    print("\n犯罪率最高的观测编号及其 crim 值：")
    print(f"编号: {max_crim_idx}, crim = {Boston.loc[max_crim_idx, 'crim']}")

    print("\n税率最高的观测编号及其 tax 值：")
    print(f"编号: {max_tax_idx}, tax = {Boston.loc[max_tax_idx, 'tax']}")

    print("\n师生比最高的观测编号及其 ptratio 值：")
    print(f"编号: {max_ptratio_idx}, ptratio = {Boston.loc[max_ptratio_idx, 'ptratio']}")

    print("\n结论：")
    print("1. 某些地区的 crim 非常高，说明犯罪率分布很不均匀。")
    print("2. tax 的范围也很大，说明地区间税率差异明显。")
    print("3. ptratio 的变化没有 crim 那么极端，但仍存在较高的地区。")

    # =========================
    # (f) 有多少郊区临 Charles River
    # =========================
    print("\n" + "=" * 60)
    print("(f) 临 Charles River 的郊区数量")
    print("=" * 60)

    chas_count = (Boston["chas"] == 1).sum()
    print(f"chas = 1 的观测个数: {chas_count}")

    # =========================
    # (g) ptratio 的中位数
    # =========================
    print("\n" + "=" * 60)
    print("(g) 城镇师生比 ptratio 的中位数")
    print("=" * 60)

    ptratio_median = Boston["ptratio"].median()
    print(f"ptratio 的中位数: {ptratio_median}")

    # =========================
    # (h) 最低 medv 的郊区及其其他变量
    # =========================
    print("\n" + "=" * 60)
    print("(h) medv 最低的观测")
    print("=" * 60)

    min_medv = Boston["medv"].min()
    min_medv_rows = Boston[Boston["medv"] == min_medv]

    print(f"最低 medv = {min_medv}")
    print("注意：数据集中没有真正的 suburb 名称，只有观测编号。")
    print("medv 最低的观测如下：")
    print(min_medv_rows)

    print("\n这些观测的其他变量与总体范围比较：")
    for idx in min_medv_rows.index:
        print(f"\n--- 观测编号 {idx} ---")
        for col in predictors.columns:
            value = Boston.loc[idx, col]
            col_min = Boston[col].min()
            col_max = Boston[col].max()
            print(f"{col}: {value}  （总体范围: {col_min} ~ {col_max}）")

    print("\n评论：")
    print("medv 最低的地区通常具有以下特征：")
    print("1. 犯罪率较高")
    print("2. 工业用地比例较高")
    print("3. 污染较高")
    print("4. 房屋较旧")
    print("5. 房间数较少")
    print("6. lstat 较高（低社会经济地位人口比例较高）")
    print("因此这些地区房价中位数最低是有一定合理性的。")

    # =========================
    # (i) rm > 7 和 rm > 8 的个数
    # =========================
    print("\n" + "=" * 60)
    print("(i) 平均房间数超过 7 和超过 8 的郊区数量")
    print("=" * 60)

    rm_gt_7 = (Boston["rm"] > 7).sum()
    rm_gt_8 = (Boston["rm"] > 8).sum()

    print(f"rm > 7 的观测个数: {rm_gt_7}")
    print(f"rm > 8 的观测个数: {rm_gt_8}")

    rm_gt_8_rows = Boston[Boston["rm"] > 8]
    print("\nrm > 8 的观测如下：")
    print(rm_gt_8_rows)

    print("\n评论：")
    print("rm > 8 的地区通常属于住房条件较好的地区。")
    print("这些地区往往房价较高，lstat 较低，整体社会经济状况较好。")

    print("\n" + "=" * 60)
    print("程序运行结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
