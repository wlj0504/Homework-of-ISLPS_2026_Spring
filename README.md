# 统计学习导论课后作业（SCUT）

2026 年春季学期，华南理工大学（SCUT）数学专业大二下学期《统计学习导论》课程相关练习与代码实现仓库。

本仓库主要用于整理我在学习 **统计学习导论** 这门课过程中完成的 Python 代码，包括基于《统计学习导论》（ISLP）相关内容的练习，也包含一个 PDF 图片提取小工具。

---

## 项目内容

目前仓库中包含以下脚本：

- `boston_exercise.py`  
  波士顿房价数据集的回归分析练习。

- `carseats_regression.py`  
  Carseats 数据集的多元线性回归与分类变量处理。

- `compare_linear_and_cubic_regression_rss.py`  
  线性回归与三次回归的 RSS 对比分析。

- `stats_learning_concepts.py`  
  统计学习基础概念相关的小脚本，用于理解分类、回归、预测与推断等内容。

- `PDF catch.py`  
  从 PDF 文件中提取高分辨率图片的小工具。

---

## 运行环境

建议使用以下环境：

- Python 3.9 及以上
- Windows / Linux / macOS 均可

---

## 依赖库

在运行本项目代码前，请先安装以下 Python 库：

### 课程练习部分依赖
- `pandas`
- `numpy`
- `statsmodels`
- `ISLP`

### PDF 图片提取工具依赖
- `pymupdf`

---

## 安装方式

可以使用以下命令安装依赖：

```bash
pip install pandas numpy statsmodels
pip install ISLP -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pymupdf
