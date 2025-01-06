def calculate_inertia(mass, size):
    # 计算惯性矩 (假设正方体)
    I = (1/6) * mass * (size**2)
    return [I, I, I]  # 对于正方体，三个方向的惯性矩是相同的

# 假设给定质量和尺寸
mass = 3  # 质量
size = 0.2  # 边长

# 计算惯性矩
inertia_values = calculate_inertia(mass, size)
print("惯性矩:", ' '.join([str(i) for i in inertia_values]))
