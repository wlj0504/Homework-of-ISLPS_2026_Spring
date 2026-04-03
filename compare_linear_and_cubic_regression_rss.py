import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# =========================================================
# 工具函数
# =========================================================
def rss(y_true, y_pred):
    """计算 RSS = sum((y - y_hat)^2)"""
    return np.sum((y_true - y_pred) ** 2)


def fit_and_evaluate(X_train, y_train, X_test, y_test):
    """
    拟合线性回归和三次回归，并返回训练/测试 RSS
    """
    # 1. 线性回归
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    y_train_pred_linear = linear_model.predict(X_train)
    y_test_pred_linear = linear_model.predict(X_test)

    train_rss_linear = rss(y_train, y_train_pred_linear)
    test_rss_linear = rss(y_test, y_test_pred_linear)

    # 2. 三次回归
    cubic_model = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        LinearRegression()
    )
    cubic_model.fit(X_train, y_train)

    y_train_pred_cubic = cubic_model.predict(X_train)
    y_test_pred_cubic = cubic_model.predict(X_test)

    train_rss_cubic = rss(y_train, y_train_pred_cubic)
    test_rss_cubic = rss(y_test, y_test_pred_cubic)

    return {
        "train_rss_linear": train_rss_linear,
        "train_rss_cubic": train_rss_cubic,
        "test_rss_linear": test_rss_linear,
        "test_rss_cubic": test_rss_cubic
    }


def simulate_case(case_name, true_function, n_train=100, n_test=1000, sigma=1.0, n_sim=500, seed=42):
    """
    重复模拟多次，比较线性回归和三次回归的训练/测试 RSS
    """
    rng = np.random.default_rng(seed)

    train_rss_linear_list = []
    train_rss_cubic_list = []
    test_rss_linear_list = []
    test_rss_cubic_list = []

    for _ in range(n_sim):
        # 训练集
        X_train = rng.uniform(-2, 2, size=(n_train, 1))
        eps_train = rng.normal(0, sigma, size=n_train)
        y_train = true_function(X_train[:, 0]) + eps_train

        # 测试集
        X_test = rng.uniform(-2, 2, size=(n_test, 1))
        eps_test = rng.normal(0, sigma, size=n_test)
        y_test = true_function(X_test[:, 0]) + eps_test

        result = fit_and_evaluate(X_train, y_train, X_test, y_test)

        train_rss_linear_list.append(result["train_rss_linear"])
        train_rss_cubic_list.append(result["train_rss_cubic"])
        test_rss_linear_list.append(result["test_rss_linear"])
        test_rss_cubic_list.append(result["test_rss_cubic"])

    print("=" * 60)
    print(f"{case_name}")
    print("=" * 60)
    print(f"平均训练 RSS（线性回归）: {np.mean(train_rss_linear_list):.4f}")
    print(f"平均训练 RSS（三次回归）: {np.mean(train_rss_cubic_list):.4f}")
    print(f"平均测试 RSS（线性回归）: {np.mean(test_rss_linear_list):.4f}")
    print(f"平均测试 RSS（三次回归）: {np.mean(test_rss_cubic_list):.4f}")

    print("\n结论：")
    if np.mean(train_rss_cubic_list) < np.mean(train_rss_linear_list):
        print("- 训练 RSS：三次回归通常更低（因为模型更灵活，至少不会比线性更差）。")
    else:
        print("- 训练 RSS：两者接近或相同。")

    if np.mean(test_rss_linear_list) < np.mean(test_rss_cubic_list):
        print("- 测试 RSS：线性回归通常更低。")
    elif np.mean(test_rss_linear_list) > np.mean(test_rss_cubic_list):
        print("- 测试 RSS：三次回归通常更低。")
    else:
        print("- 测试 RSS：两者非常接近。")

    print()


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    # -----------------------------------------------------
    # 情形1：真实关系是线性的
    # Y = beta0 + beta1 * X + epsilon
    # -----------------------------------------------------
    beta0 = 2.0
    beta1 = 3.0


    def true_linear(x):
        return beta0 + beta1 * x


    simulate_case(
        case_name="情形1：真实关系是线性",
        true_function=true_linear,
        n_train=100,
        n_test=1000,
        sigma=1.0,
        n_sim=500,
        seed=1
    )

    # -----------------------------------------------------
    # 情形2：真实关系不是线性的（设成三次）
    # Y = beta0 + beta1*X + beta2*X^2 + beta3*X^3 + epsilon
    # -----------------------------------------------------
    beta0 = 2.0
    beta1 = 3.0
    beta2 = -2.0
    beta3 = 1.5


    def true_nonlinear(x):
        return beta0 + beta1 * x + beta2 * x ** 2 + beta3 * x ** 3


    simulate_case(
        case_name="情形2：真实关系是明显非线性（三次）",
        true_function=true_nonlinear,
        n_train=100,
        n_test=1000,
        sigma=1.0,
        n_sim=500,
        seed=2
    )

    # -----------------------------------------------------
    # 情形3：真实关系非线性，但离线性“远近不确定”
    # 通过调节非线性强度 gamma 来观察测试 RSS 的变化
    # -----------------------------------------------------
    print("=" * 60)
    print("情形3：真实关系非线性，但不知道偏离线性有多远")
    print("=" * 60)

    gammas = [0.0, 0.2, 0.5, 1.0, 2.0]

    for gamma in gammas:
        def true_partial_nonlinear(x, g=gamma):
            return 2.0 + 3.0 * x + g * (x ** 2 - 0.5 * x ** 3)


        rng = np.random.default_rng(123 + int(gamma * 10))
        train_linear_all = []
        train_cubic_all = []
        test_linear_all = []
        test_cubic_all = []

        for _ in range(300):
            X_train = rng.uniform(-2, 2, size=(100, 1))
            y_train = true_partial_nonlinear(X_train[:, 0]) + rng.normal(0, 1.0, size=100)

            X_test = rng.uniform(-2, 2, size=(1000, 1))
            y_test = true_partial_nonlinear(X_test[:, 0]) + rng.normal(0, 1.0, size=1000)

            result = fit_and_evaluate(X_train, y_train, X_test, y_test)

            train_linear_all.append(result["train_rss_linear"])
            train_cubic_all.append(result["train_rss_cubic"])
            test_linear_all.append(result["test_rss_linear"])
            test_cubic_all.append(result["test_rss_cubic"])

        print(f"\n非线性强度 gamma = {gamma}")
        print(f"平均训练 RSS（线性）: {np.mean(train_linear_all):.4f}")
        print(f"平均训练 RSS（三次）: {np.mean(train_cubic_all):.4f}")
        print(f"平均测试 RSS（线性）: {np.mean(test_linear_all):.4f}")
        print(f"平均测试 RSS（三次）: {np.mean(test_cubic_all):.4f}")

    print("\n程序说明：")
    print("1. 对于训练 RSS，三次回归一般都会小于或等于线性回归。")
    print("2. 如果真实关系真的是线性，测试 RSS 往往线性回归更小。")
    print("3. 如果真实关系不是线性，训练 RSS 仍通常是三次回归更小。")
    print("4. 对于测试 RSS，在非线性情形下谁更小取决于真实函数离线性有多远，因此“不能一概而论”。")
