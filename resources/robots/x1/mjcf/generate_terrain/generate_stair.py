num_steps = 10  # 台阶数量
step_height = 0.03  # 每级台阶的高度
step_depth = 0.2  # 每级台阶的深度

# 上楼梯
for i in range(num_steps):
    print(f'<geom name="step_up{i+1}" type="box" size="0.5 2 {step_height/2}" pos="{i*step_depth+2} 0 {step_height*(i+1)}" material="stair_material" conaffinity="7" condim="3" friction="1"/>')

# 下楼梯
for i in range(num_steps):
    print(f'<geom name="step_down{i+1}" type="box" size="0.5 2 {step_height/2}" pos="{num_steps*step_depth + 2 + i*step_depth} 0 {step_height*(num_steps - i - 1)}" material="stair_material" conaffinity="7" condim="3" friction="1"/>')
