(defun c:EXPLODE_BLOCKS_BY_LAYER_SAFE_FINAL_V2
  (/ protected_layers ss_all_blocks ss_to_explode blk_ent blk_layer
     prot skipped_count exploded_count i
     target-folder dwgname newname new-fullpath)

  (vl-load-com)
  (princ "\n🚀 开始安全炸开操作...")

  ;; === Step 1: 解锁所有图层 ===
  (princ "\n🔓 解锁所有图层中...")
  (vl-cmdf "_.LAYER" "_U" "*" "_ON" "*" "_THAW" "*" "")
  (princ "\n✅ 图层全部解锁。")

  ;; === Step 2: 选择所有块 ===
  (setq ss_all_blocks (ssget "_X" '((0 . "INSERT"))))
  (if (not ss_all_blocks)
    (progn
      (princ "\n⚠ 未检测到任何块引用，退出。")
      (princ)
      (exit)
    )
  )

  ;; === Step 3: 筛选需要炸开的块 ===
  (setq protected_layers
    '("*WALL*" "*WIN*" "*DOOR*" "*COL*" "*窗*" "*门*" "*柱*"
      "*标注*" "*TEXT*" "*NOTE*" "*SYMB*" "*符号*" "*ROOM*" "*RMNAME*"))
  (setq ss_to_explode (ssadd))
  (setq skipped_count 0)

  (setq i 0)
  (repeat (sslength ss_all_blocks)
    (setq blk_ent (ssname ss_all_blocks i))
    (setq blk_layer (strcase (cdr (assoc 8 (entget blk_ent)))))
    (setq prot nil)
    (foreach p protected_layers
      (if (wcmatch blk_layer (strcase p)) (setq prot T))
    )
    (if (not prot)
      (ssadd blk_ent ss_to_explode)
      (setq skipped_count (1+ skipped_count))
    )
    (setq i (1+ i))
  )

  (princ (strcat "\n🔍 找到 " (itoa (sslength ss_to_explode)) " 个可炸开的块，跳过 " (itoa skipped_count) " 个。"))

  ;; === Step 4: 执行炸开 ===
  (setq exploded_count 0)
  (if (> (sslength ss_to_explode) 0)
    (progn
      (setq i 0)
      (repeat (sslength ss_to_explode)
        (setq blk_ent (ssname ss_to_explode i))
        (if (and blk_ent (entget blk_ent))
          (progn
            (vl-catch-all-apply 'vl-cmdf (list "_.EXPLODE" blk_ent))
            (setq exploded_count (1+ exploded_count))
          )
        )
        (setq i (1+ i))
      )
      (princ (strcat "\n✅ 已炸开约 " (itoa exploded_count) " 个块。"))
    )
    (princ "\nℹ 没有可炸开的块。")
  )

  ;; === Step 5: 导出副本 ===
  (setq target-folder (strcat (getenv "USERPROFILE") "\\Desktop\\Exploded_DWGs"))
  (if (not (vl-file-directory-p target-folder))
    (vl-mkdir target-folder))

  (setq dwgname (getvar "DWGNAME"))
  (setq newname (strcat (vl-filename-base dwgname) "_exploded.dwg"))
  (setq new-fullpath (strcat target-folder "\\" newname))

  (princ (strcat "\n💾 正在保存副本到: " new-fullpath))
  (command "_.UNDO" "_Begin")
  (command "_.WBLOCK" new-fullpath "" "0,0,0" "_ALL" "")
  (command "_.UNDO" "_End")
  (command "_.U")

  (princ "\n🎯 当前文件未被修改，副本已生成。")
  (princ)
)
