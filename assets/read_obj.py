import trimesh
import matplotlib.pyplot as plt

# 加载模型
mesh = trimesh.load('assets/building/building.obj')

# 获取模型顶点坐标投影（简单显示）
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制三角网格
ax.plot_trisurf(
    mesh.vertices[:, 0],
    mesh.vertices[:, 1],
    mesh.faces,
    mesh.vertices[:, 2],
    color='lightgray',
    edgecolor='gray',
    linewidth=0.2,
    alpha=1.0
)

# 添加坐标轴标签和刻度
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 可选：设置坐标轴范围（根据模型大小自动适配也可）
ax.auto_scale_xyz(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2])

# 可选：去掉网格线（保留坐标轴）
ax.grid(False)

plt.tight_layout()
plt.savefig("output_image_with_axes.png", dpi=300)
plt.show()



# 提取 XY 坐标（投影到 XY 平面）
x = mesh.vertices[:, 0]
y = mesh.vertices[:, 1]
faces = mesh.faces

# 创建 2D 图像
plt.figure(figsize=(6, 6))
plt.triplot(x, y, faces, color='gray', linewidth=0.5)

# 设置坐标轴标签和等比例显示
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal')
plt.title('2D Projection of Mesh (XY plane)')
plt.tight_layout()
plt.savefig("projection_xy_2d.png", dpi=300)
plt.show()