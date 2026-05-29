# 5.1 量化技术基础：精度体系与线性量化

本节介绍FP32/FP16/BF16/INT8/INT4等精度体系，
以及线性量化的核心公式：x_q = round(x/scale) - zero_point。