# -*- coding: utf-8 -*-

import rhinoscriptsyntax as rs
import math

# ==============================================================================
# 1. 配置区域
# ==============================================================================

room_categories = {
    "核心办公空间": {
        "keywords": ["办公室", "办公", "总经理", "董事长", "总裁", "主任", "总监", "财务", "人事", "行政", "设计", "研发", "项目部", "工位"],
        "layer": "核心办公空间"
    },
    "公共协作空间": {
        "keywords": ["会议", "多功能", "展厅", "洽谈", "接待", "休息", "茶室", "咖啡", "活动", "路演", "分享区", "头脑风暴"],
        "layer": "公共协作空间"
    },
    "交通联系空间": {
        "keywords": ["走廊", "走道", "过道", "过厅", "前室", "门厅", "大堂", "大厅", "电梯厅", "楼梯", "DT", "上", "下"],
        "layer": "交通联系空间"
    },
    "服务支持空间": {
        "keywords": ["卫生间", "厕所", "洗手间", "WC", "卫", "茶水", "打印", "复印", "储藏", "库房", "清洁", "保洁", "机房", "设备", "强电", "弱电", "配电", "管井", "水井", "电井", "通信", "排烟", "排风", "正压送风", "水管井", "消防控制", "空调机房"],
        "layer": "服务支持空间"
    }
}

WALL_LAYER_NAME = "墙体"

# ==============================================================================
# 2. 功能函数库
# ==============================================================================

def prepare_layers(categories, wall_layer, parent_layer="空间分类"):
    if not rs.IsLayer(parent_layer): rs.AddLayer(parent_layer)
    for category in categories.values():
        full_layer_path = "{}::{}".format(parent_layer, category["layer"])
        if not rs.IsLayer(full_layer_path): rs.AddLayer(full_layer_path)
    wall_full_path = "{}::{}".format(parent_layer, wall_layer)
    if not rs.IsLayer(wall_full_path): rs.AddLayer(wall_full_path)
    print("图层结构准备就绪。")

def preprocess_and_create_regions_user_specified():
    """
    【按您的要求更新】
    对曲线进行预处理（炸开、删重）并生成平面区域。
    """
    print("开始预处理曲线 (Explode, SelDup)...")
    
    all_curves = rs.ObjectsByType(4)
    if not all_curves:
        print("警告：模型中未找到任何曲线。")
        return None
    rs.SelectObjects(all_curves)

    rs.Command("_-Explode", False)
    
    # 【关键修复】根据您的发现，在 Explode 后取消全选，确保 SelDup 从干净的状态开始
    rs.UnselectAllObjects()

    rs.Command("_-SelDup", False)
    
    duplicates = rs.SelectedObjects()
    if duplicates:
        rs.Command("_-Delete", False)
        print("删除了 {} 个重复曲线。".format(len(duplicates)))
    else:
        print("未发现重复曲线。")
    
    rs.UnselectAllObjects()
    
    print("正在从处理后的曲线生成平面区域...")
    curves_for_region = rs.ObjectsByType(4)
    if not curves_for_region:
        print("错误：预处理后没有可用于生成区域的曲线。")
        return None

    rs.SelectObjects(curves_for_region)

    if rs.Command("_TestGetPlanarRegions", True):
        all_surfaces = rs.LastCreatedObjects(select=False)
        if all_surfaces:
            print("成功生成 {} 个曲面。".format(len(all_surfaces)))
            rs.UnselectAllObjects()
            return all_surfaces

    print("错误：未能从曲线生成任何曲面。")
    rs.UnselectAllObjects()
    return None

def classify_text_and_get_points(categories):
    all_text_objects = [obj for obj in rs.AllObjects() if rs.IsText(obj)]
    classified_points = {category: [] for category in categories}
    for text_obj in all_text_objects:
        text_string = rs.TextObjectText(text_obj)
        if not text_string: continue
        for category, data in categories.items():
            sorted_keywords = sorted(data["keywords"], key=len, reverse=True)
            for keyword in sorted_keywords:
                if keyword in text_string:
                    point = rs.TextObjectPoint(text_obj)
                    classified_points[category].append(point)
                    goto_next_text = True
                    break
            else:
                goto_next_text = False
                continue
            if goto_next_text:
                break
    return classified_points

def calculate_compactness(srf):
    try:
        area = rs.SurfaceArea(srf)[0]
        edges = rs.DuplicateEdgeCurves(srf)
        if not edges: return 0, 0
        perimeter = sum(rs.CurveLength(edge) for edge in edges)
        rs.DeleteObjects(edges)
        if perimeter <= 0: return 0, area
        compactness = (4 * math.pi * area) / (perimeter**2)
        return compactness, area
    except Exception:
        return 0, 0

# ==============================================================================
# 3. 主执行函数
# ==============================================================================

def main():
    rs.UnselectAllObjects()
    
    PARENT_LAYER = "空间分类"
    prepare_layers(room_categories, WALL_LAYER_NAME, PARENT_LAYER)
    
    all_surfaces = preprocess_and_create_regions_user_specified()
    if not all_surfaces:
        print("由于未能生成曲面，脚本终止。")
        return
        
    classified_surfaces = []
    remaining_surfaces = list(all_surfaces)

    print("\n--- 开始第一阶段：房间分类 ---")
    classified_points = classify_text_and_get_points(room_categories)
    rooms_found_count = 0
    surfaces_for_next_stage = []
    
    for surface_id in remaining_surfaces:
        is_matched = False
        for category, points in classified_points.items():
            for point in points:
                if rs.IsPointOnSurface(surface_id, point):
                    layer_path = "{}::{}".format(PARENT_LAYER, room_categories[category]["layer"])
                    rs.ObjectLayer(surface_id, layer_path)
                    classified_surfaces.append(surface_id)
                    rooms_found_count += 1
                    is_matched = True
                    break
            if is_matched:
                break
        if not is_matched:
            surfaces_for_next_stage.append(surface_id)

    print("房间分类完成，共识别了 {} 个房间。".format(rooms_found_count))
    remaining_surfaces = surfaces_for_next_stage

    print("\n--- 开始第二阶段：墙体筛选 ---")
    walls_found_count = 0
    wall_layer_path = "{}::{}".format(PARENT_LAYER, WALL_LAYER_NAME)
    unclassified_final = []

    for srf in remaining_surfaces:
        compactness, area = calculate_compactness(srf)
        is_wall = (area < 2) or (compactness < 0.4 and area < 10)
        
        if is_wall:
            rs.ObjectLayer(srf, wall_layer_path)
            classified_surfaces.append(srf)
            walls_found_count += 1
        else:
            unclassified_final.append(srf)

    print("墙体筛选完成，共识别了 {} 个墙体图块。".format(walls_found_count))

    print("\n--- 开始最后阶段：状态管理 ---")
    if classified_surfaces:
        rs.LockObjects(classified_surfaces)
        print("已锁定 {} 个已分类的房间和墙体对象。".format(len(classified_surfaces)))
    
    if unclassified_final:
        rs.SelectObjects(unclassified_final)
        print("留下了 {} 个未分类图块供手动处理（已选中）。".format(len(unclassified_final)))
    else:
        print("所有图块均已成功分类。")

    print("\n脚本执行完毕！")

# ==============================================================================
# 4. 脚本入口
# ==============================================================================

if __name__ == "__main__":
    main()