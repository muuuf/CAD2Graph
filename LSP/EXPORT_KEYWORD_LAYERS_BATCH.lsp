;;; --- 最终解决方案版本：集成完整关键词字典 + 使用UNDO命令 ---
(defun c:EXPORT_KEYWORD_LAYERS_BATCH (/ keywords ss_all key ss_temp export_path)
  
  ;; --- ★★★ 最终版关键词字典 ★★★ ---
  (setq keywords 
    '(
      ; --- 1. 核心建筑构件 (Architecture) ---
      "*WALL*" "*墙*"
      "*WIN*" "*WNDW*" "*DOOR*" "*窗*" "*门*"
      "*COL*" "*柱*"
      "*CURTAIN*" "*CURTWALL*" "*幕墙*"

      ; --- 2. 注释与符号 (Annotation) ---
      "*标注*"
      "*TEXT*" "*NOTE*" "*文字*"
      "*SYMB*" "*符号*"
      "*ROOM*" "*RMNAME*"
    )
  )
  ;; --- ★★★ 字典结束 ★★★ ---
  
  (setq ss_all (ssadd)) ; 创建一个空的总选择集
  
  ;; 遍历所有关键词，将找到的对象添加到总选择集中
  (foreach key keywords
    (if (setq ss_temp (ssget "_X" (list (cons 8 key))))
      (progn
        (repeat (setq i (sslength ss_temp))
          (setq ss_all (ssadd (ssname ss_temp (setq i (1- i))) ss_all))
        )
      )
    )
  )
  
  ;; 检查是否找到了任何对象
  (if (> (sslength ss_all) 0)
    (progn
      ;; 准备导出路径
      (setq export_path (strcat (getvar "dwgprefix") (vl-filename-base (getvar "dwgname")) "_filtered.dwg"))
      (if (findfile export_path) (vl-file-delete export_path)) ; 如果文件已存在则删除
      
      ;; 为了防止WBLOCK删除源对象，我们执行“导出-撤销”操作
      (command "_.UNDO" "_Begin")
      (command "_.WBLOCK" export_path "" "0,0,0" ss_all "")
      (command "_.UNDO" "_End")
      (command "_.U")
      
      (princ (strcat "\nSuccessfully exported " (itoa (sslength ss_all)) " objects to: " export_path))
    )
    (princ "\nNo objects found with the specified keywords.")
  )
  (princ) ; 安静地退出命令
)