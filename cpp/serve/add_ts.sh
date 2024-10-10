#!/bin/bash

# 檢查是否有提供資料夾路徑
if [ -z "$1" ]; then
    echo "請提供資料夾路徑。"
    exit 1
fi

# 設定資料夾路徑
FOLDER_PATH=$1

# 在資料夾內搜尋所有 .cc 和 .cpp 文件
find "$FOLDER_PATH" -type f \( -name "*.cc" -o -name "*.cpp" \) | while read -r file; do
    # 暫存匹配到的行號和內容
    matches=()
    while IFS= read -r line; do
        matches+=("$line")
    done < <(grep -n "NVTXScopedRange" "$file")
    
    # 從最後一個匹配的行開始處理
    for ((i=${#matches[@]}-1; i>=0; i--)); do
        line="${matches[$i]}"
        # 取得行號
        line_number=$(echo "$line" | cut -d: -f1)
        # 提取 nvtx_scope 的內容作為前綴
        prefix=$(echo "$line" | sed -n 's/.*NVTXScopedRange[^"]*("\([^"]*\)").*/\1/p')

        # 檢查是否成功提取前綴
        if [ -n "$prefix" ]; then
            # 計算 NVTXScopedRange 行的下一行行號
            next_line_number=$((line_number + 1))
            # 插入 C++ 代碼到下一行
            sed -i "${next_line_number}i auto now = std::chrono::system_clock::now();\\
std::time_t timestamp = std::chrono::system_clock::to_time_t(now);\\
LOG(INFO) << \"${prefix}, timestamp: \" << timestamp << std::endl;" "$file"
        fi
    done
done

echo "完成在每個 .cc 和 .cpp 文件中插入帶有前綴的 C++ 代碼來印出 timestamp。"

