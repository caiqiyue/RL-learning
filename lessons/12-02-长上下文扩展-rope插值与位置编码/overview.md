# 12.2 长上下文扩展：RoPE插值与位置编码

本节讲解RoPE的旋转编码原理，以及Linear/NTK-aware/YaRN等
位置插值方法如何将模型上下文窗口从4K扩展到32K+。