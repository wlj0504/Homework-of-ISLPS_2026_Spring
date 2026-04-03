# -*- coding: utf-8 -*-

def analyze_scenario(scenario_name, target_type, goal, n, p, suggested_models):
    """
    打印并分析给定的机器学习场景
    """
    print(f"--- 场景 {scenario_name} ---")

    if target_type == "Quantitative":
        problem_type = "回归 (Regression)"
    else:
        problem_type = "分类 (Classification)"

    print(f"1. 问题类型: {problem_type}")
    print(f"2. 核心目标: {goal}")
    print(f"3. 样本量 n: {n}")
    print(f"4. 预测变量数 p: {p}")
    print(f"5. Python中常用的模型/库: {suggested_models}\n")


def main():
    print("ISLR / ISLP Chapter 2: Exercise 2 解答\n" + "=" * 40 + "\n")

    # 场景 (a)：研究影响 CEO 薪水的因素
    # 分析：CEO 薪水是连续的定量数值（回归）；目的是“understanding which factors affect”（推断）；
    # 500家公司（n=500）；利润、员工数、行业 3 个特征（p=3）。
    analyze_scenario(
        scenario_name="(a) CEO 薪水影响因素分析",
        target_type="Quantitative",
        goal="推断 (Inference) - 重点在于解释哪些特征对目标变量有影响",
        n=500,
        p=3,  # profit, number of employees, industry
        suggested_models="statsmodels.api.OLS (普通最小二乘法)"
    )

    # 场景 (b)：预测新产品是成功还是失败
    # 分析：成功/失败是定性的分类标签（分类）；目的是“wish to know whether it will be a success or a failure”（预测）；
    # 20个历史产品（n=20）；价格、营销预算、竞品价格 + 其他10个变量（p=13）。
    analyze_scenario(
        scenario_name="(b) 新产品成败预测",
        target_type="Qualitative",
        goal="预测 (Prediction) - 重点在于利用历史数据对未知的新产品打上准确的标签",
        n=20,
        p=13,  # price, marketing budget, competition price + 10 other variables
        suggested_models="sklearn.linear_model.LogisticRegression, sklearn.ensemble.RandomForestClassifier"
    )

    # 场景 (c)：预测美元/欧元汇率百分比变化
    # 分析：汇率变动百分比是连续的定量数值（回归）；目的是“interested in predicting”（预测）；
    # 2012年全年的周度数据，一年有52周（n=52）；美国、英国、德国市场的变化率（p=3）。
    analyze_scenario(
        scenario_name="(c) 美元/欧元汇率变化预测",
        target_type="Quantitative",
        goal="预测 (Prediction) - 重点在于得到尽可能准确的汇率变动数值",
        n=52,  # 2012年的总周数
        p=3,  # US market %, British market %, German market %
        suggested_models="sklearn.linear_model.Ridge, sklearn.svm.SVR, 或者时间序列模型 (statsmodels.tsa.arima.model.ARIMA)"
    )


if __name__ == "__main__":
    main()
