# -*- coding: utf-8 -*-

import rhinoscriptsyntax as rs
import scriptcontext as sc
import System.Drawing.Color as Color

def get_text_coordinates():
    # 获取所有对象
    all_objects = rs.AllObjects()
    
    # 筛选出所有文本对象
    text_objects = [obj for obj in all_objects if rs.IsText(obj)]
    
    # 获取每个文本对象的坐标和内容，并存储在字典中
    text_dict = {}
    for text_obj in text_objects:
        text_string = rs.TextObjectText(text_obj)
        text_position = rs.TextObjectPoint(text_obj)
        
        # 使用字符串格式化创建唯一键
        unique_key = "{}_{}".format(text_string, text_position)
        text_dict[unique_key] = (text_string, text_position)
    
    return text_dict

def search_text_coordinates(text_dict, keyword):
    # 搜索包含关键字的文本对象
    result = {key: value for key, value in text_dict.items() if keyword in key}
    return result

def select_all_curves():
    all_objects = rs.AllObjects()
    curve_ids = [obj for obj in all_objects if rs.IsCurve(obj)]
    return curve_ids

def main(keyword, color):
    # 创建字典
    text_dict = get_text_coordinates()
    
    if not text_dict:
        print("No text objects found.")
        return

    # 进行关键字搜索
    search_result = search_text_coordinates(text_dict, keyword)
    keyword_points = []
    
    if search_result:
        for key, (text, coord) in search_result.items():
            keyword_points.append(coord)
    else:
        print("No text objects found with keyword '{}'.".format(keyword))
        return

    # 输出调试信息
    print("Found keyword points:", keyword_points)

    # 自动选择所有曲线
    curve_ids = select_all_curves()
    if not curve_ids:
        print("未找到任何曲线")
        return
    
    # 运行 _TestGetPlanarRegions 命令
    command_result = rs.Command("_TestGetPlanarRegions", False)
    if not command_result:
        print("_TestGetPlanarRegions 命令执行失败")
        return
    
    # 获取生成的曲面
    target_surfaces = rs.LastCreatedObjects(select=False)
    if not target_surfaces:
        print("未生成有效的曲面")
        return

    # 过滤包含点的曲面
    valid_surfaces = []
    for surface_id in target_surfaces:
        for point in keyword_points:
            if rs.IsPointOnSurface(surface_id, point):
                valid_surfaces.append(surface_id)
                break
    
    if not valid_surfaces:
        print("没有曲面包含检测点")
        return

    # 输出调试信息
    print("Valid surfaces:", valid_surfaces)

    # 删除不在 valid_surfaces 列表中的面
    surfaces_to_delete = [surface_id for surface_id in target_surfaces if surface_id not in valid_surfaces]
    if surfaces_to_delete:
        rs.DeleteObjects(surfaces_to_delete)
        print("已删除无关的曲面")

    # 为有效的面设置颜色
    for surface_id in valid_surfaces:
        rs.ObjectColor(surface_id, color)
    
    # 选择包含检测点的曲面
    rs.SelectObjects(valid_surfaces)
    print("已选择包含检测点的曲面并设置颜色")

if __name__ == "__main__":
    main('展厅', Color.Red)