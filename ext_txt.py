import os
import shutil

NIR_LIST = "polyu__NIR_list.txt"   
RED_LIST = "polyu__Red_list.txt"   

OUT_ROOT = "dataset/test"       
OUT_IR   = os.path.join(OUT_ROOT, "ir")
OUT_VI   = os.path.join(OUT_ROOT, "vi")

OUT_LIST = "phase2_test_pairs.txt" 

def parse_test_list(list_file):
    """
    解析 txt，返回：
    dict[(pid_str, filename)] = (img_path, label_int)
    
    只保留 split == 'test' 的样本。
    """
    result = {}
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # 正常行应该是：path label split
            if len(parts) != 3:
                continue
            img_path, label_str, split = parts
            if split != "test":
                continue

            label = int(label_str)
            # pid 是倒数第二级目录名，例如 ...\NIR\0086\2_02_s.jpg -> pid = '0086'
            pid = os.path.basename(os.path.dirname(img_path))
            filename = os.path.basename(img_path)

            result[(pid, filename)] = (img_path, label)
    return result


def main():
    os.makedirs(OUT_IR, exist_ok=True)
    os.makedirs(OUT_VI, exist_ok=True)

    # 1. 解析 NIR 测试样本
    nir_test = parse_test_list(NIR_LIST)
    print(f"NIR test samples: {len(nir_test)}")

    # 2. 解析 Red 测试样本并与 NIR 对齐
    pairs = []
    counter = {}  # 统计每个 (label, side) 的编号：((label, 'L'/'R') -> idx)

    with open(RED_LIST, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            red_path, label_str, split = parts
            if split != "test":
                continue

            label_red = int(label_str)
            pid = os.path.basename(os.path.dirname(red_path))
            filename = os.path.basename(red_path)
            key = (pid, filename)

            # 必须 NIR/Red 同一个人、同一个文件名都在 test 才能配成一对
            if key not in nir_test:
                continue

            nir_path, label_nir = nir_test[key]
            assert label_red == label_nir, "NIR / Red 标签不一致！"

            label = label_nir  # 0-based

            # 从文件名里判断左右手：例如 "1_02_s.jpg" 开头的 1 / 2
            name_no_ext = os.path.splitext(filename)[0]  # 1_02_s
            first_token = name_no_ext.split("_")[0]      # "1" or "2"
            side = "L" if first_token == "1" else "R"    # 约定：1->L, 2->R

            # 人 ID：label 是 0 开始，文件名中希望是 001, 002...
            person_id = label + 1
            person_str = f"{person_id:03d}"

            # 第几张：针对同一个 (label, side) 自增
            idx = counter.get((label, side), 0) + 1
            counter[(label, side)] = idx

            new_name = f"{person_str}_{side}_{idx}.jpg"

            # 目标路径（真实复制的文件）
            dst_ir = os.path.join(OUT_IR, new_name)
            dst_vi = os.path.join(OUT_VI, new_name)

            # 复制文件
            shutil.copy2(nir_path, dst_ir)
            shutil.copy2(red_path, dst_vi)

            # 写入列表时，使用相对路径（和你需求里的格式一致）
            rel_ir = os.path.join("dataset", "test", "ir", new_name).replace("\\", "/")
            rel_vi = os.path.join("dataset", "test", "vi", new_name).replace("\\", "/")

            pairs.append(f"{rel_ir} {rel_vi} {label}\n")

    # 3. 写出测试用 pairs 列表
    with open(OUT_LIST, "w", encoding="utf-8") as f:
        f.writelines(pairs)

    print(f"Total paired test samples: {len(pairs)}")
    print(f"Test list saved to: {OUT_LIST}")
    print(f"Images copied to:\n  {OUT_IR}\n  {OUT_VI}")


if __name__ == "__main__":
    main()
