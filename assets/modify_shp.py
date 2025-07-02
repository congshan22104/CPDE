import geopandas as gpd
import matplotlib.pyplot as plt

# 读取 .shp 文件（请将路径替换为你的文件路径）
shapefile_path = 'assets/shp/buliding.shp'
gdf = gpd.read_file(shapefile_path)

# 打印前几行属性数据
print(gdf.head())
# 打印数据的边界范围（最小值和最大值）
print("数据的最小值和最大值：", gdf.total_bounds)


# 定义经纬度范围（例如：xmin, ymin, xmax, ymax）
xmin, ymin, xmax, ymax = 800500.0, 2505000 ,802000.0, 2506000  # 设定你需要的经纬度范围

# 过滤数据：根据坐标范围筛选
gdf_filtered = gdf.cx[xmin:xmax, ymin:ymax]

# 打印过滤后的数据
print(gdf_filtered.head())

# 绘制过滤后的地图
gdf_filtered.plot(edgecolor='black', figsize=(8, 8))
plt.title("Filtered Shapefile Visualization")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis('equal')  # 保持坐标比例
plt.tight_layout()
plt.savefig("filtered_shapefile_plot.png", dpi=300)
plt.show()

# 保存过滤后的数据到新文件（例如：filtered_building.shp）
filtered_shapefile_path = 'assets/shp/filtered_building.shp'
gdf_filtered.to_file(filtered_shapefile_path)
