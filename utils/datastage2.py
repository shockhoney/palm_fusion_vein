import os

def parse_list_file(file_path):
    """
    解析列表文件，返回一个字典。
    Key: (label, filename) -> (img_full_path, split)
    """
    data_map = {}
    print(f"正在读取文件: {file_path} ...")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return data_map

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            # 格式: path label split
            path = parts[0]
            label = parts[1]
            split = parts[2]
            
            # 获取文件名 (例如: 1_01_s.jpg) 用于匹配
            filename = os.path.basename(path)
            
            # 统一路径分隔符，避免Windows/Linux兼容问题
            path = path.replace("\\", "/")
            
            #以此作为唯一键值进行匹配
            key = (label, filename)
            data_map[key] = (path, split)
            
    return data_map

def generate_phase2_lists(palm_file, vein_file, out_train, out_val, out_test=None):
    # 1. 读取两个文件
    palm_data = parse_list_file(palm_file)
    vein_data = parse_list_file(vein_file)
    
    if not palm_data or not vein_data:
        print("数据读取失败，请检查路径。")
        return

    pairs_train = []
    pairs_val = []
    pairs_test = []

    # 2. 遍历 Palm 数据，寻找对应的 Vein 数据
    count = 0
    for key, (palm_path, palm_split) in palm_data.items():
        if key in vein_data:
            vein_path, vein_split = vein_data[key]
            
            # 确保划分(train/val/test)是一致的
            if palm_split != vein_split:
                continue
                
            label = key[0]
            
            # Phase 2 格式: palm_path vein_path label
            line = f"{palm_path} {vein_path} {label}\n"
            
            if palm_split == 'train':
                pairs_train.append(line)
            elif palm_split == 'val':
                pairs_val.append(line)
            elif palm_split == 'test':
                pairs_test.append(line)
            
            count += 1

    # 3. 写入文件
    with open(out_train, 'w', encoding='utf-8') as f:
        f.writelines(pairs_train)
        
    with open(out_val, 'w', encoding='utf-8') as f:
        f.writelines(pairs_val)

    # 如果需要test也生成出来（虽然train.py里可能暂时不用）
    if out_test:
        with open(out_test, 'w', encoding='utf-8') as f:
            f.writelines(pairs_test)

    print("-" * 30)
    print(f"配对完成! 总共找到 {count} 对图片。")
    print(f"生成的训练集 (Train): {len(pairs_train)} 对 -> {out_train}")
    print(f"生成的验证集 (Val)  : {len(pairs_val)} 对 -> {out_val}")
    if out_test:
        print(f"生成的测试集 (Test) : {len(pairs_test)} 对 -> {out_test}")
    print("-" * 30)

if __name__ == '__main__':
    # 配置文件路径
    PALM_LIST = 'polyu_Red_list.txt'
    VEIN_LIST = 'polyu_NIR_list.txt'
    
    # 输出文件路径
    OUTPUT_TRAIN = 'polyu_phase2_train.txt'
    OUTPUT_VAL   = 'polyu_phase2_val.txt'
    OUTPUT_TEST  = 'polyu_phase2_test.txt' # 可选

    generate_phase2_lists(PALM_LIST, VEIN_LIST, OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST)
