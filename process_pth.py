import torch
import numpy as np
from plyfile import PlyData, PlyElement

# 加载数据
data = torch.load("outputs/bouncingballs_test_node/deform/iteration_9999/deform.pth", map_location="cpu")

# import pdb
# pdb.set_trace()

xyz = data["gs__xyz"].squeeze(1).detach().cpu().numpy()            # [N, 3]
colors = data["gs__features_dc"].squeeze(1).detach().cpu().numpy() # [N, 3]
colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

# 如果有透明度
if "gs__opacity" in data:
    alpha = data["gs__opacity"].squeeze(1).detach().cpu().numpy()
    alpha = (np.clip(alpha, 0, 1) * 255).astype(np.uint8).flatten()
else:
    alpha = np.full((xyz.shape[0],), 255, dtype=np.uint8)

# 处理标签数据
if "gs__label" in data:
    labels = data["gs__label"].detach().cpu().numpy()  # [N]
    print(f"Found labels: {len(labels)} Gaussians")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # 定义标签颜色映射
    label_colors = {
        0: [128, 128, 128],  # 背景 - 灰色
        1: [255, 0, 0],      # 球1 - 红色
        2: [0, 255, 0],      # 球2 - 绿色
        3: [0, 0, 255],      # 球3 - 蓝色
    }
    
    # 根据标签重新着色
    label_colors_array = np.array([label_colors.get(label, [128, 128, 128]) for label in labels])
    colors = label_colors_array.astype(np.uint8)
    print("Applied label-based coloring")
else:
    labels = np.zeros(xyz.shape[0], dtype=np.int32)  # 默认标签为0
    print("No labels found, using original colors")

# 构造 ply 顶点数据
vertices = np.array([
    (x, y, z, r, g, b, a, label)
    for (x, y, z), (r, g, b), a, label in zip(xyz, colors, alpha, labels)
], dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'), ('label', 'i4')
])

# 写入 ply 文件
ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
ply.write('outputs/bouncingballs_test_node/control_points.ply')
print("✅ Exported to output_gaussians.ply")
