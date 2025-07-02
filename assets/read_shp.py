import geopandas as gpd
import matplotlib.pyplot as plt

# 读取 .shp 文件（请将路径替换为你的文件路径）
shapefile_path = 'assets/shp/buliding.shp'
gdf = gpd.read_file(shapefile_path)

# 打印前几行属性数据
print(gdf.head())

# 绘制地图
gdf.plot(edgecolor='black', figsize=(8, 8))
plt.title("Shapefile Visualization")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis('equal')  # 保持坐标比例
plt.tight_layout()
plt.savefig("shapefile_plot.png", dpi=300)
plt.show()
