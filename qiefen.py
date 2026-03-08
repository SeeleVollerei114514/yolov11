import shutil
import random
import os

# 原始路径（确保以斜杠结尾）
image_original_path = "D:/ultralytics-8.3.33(yolo)/data/images/"
label_original_path = "D:/ultralytics-8.3.33(yolo)/data/labels/"

cur_path = os.getcwd()
# 训练集路径
train_image_path = os.path.join(cur_path, "datasets/images/train/")
train_label_path = os.path.join(cur_path, "datasets/labels/train/")

# 验证集路径
val_image_path = os.path.join(cur_path, "datasets/images/val/")
val_label_path = os.path.join(cur_path, "datasets/labels/val/")

# 测试集路径
test_image_path = os.path.join(cur_path, "datasets/images/test/")
test_label_path = os.path.join(cur_path, "datasets/labels/test/")

# 训练集目录
list_train = os.path.join(cur_path, "datasets/train.txt")
list_val = os.path.join(cur_path, "datasets/val.txt")
list_test = os.path.join(cur_path, "datasets/test.txt")

train_percent = 0.8
val_percent = 0.1
test_percent = 0.1


def del_file(path):
    """清空目录但不删除目录本身"""
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"删除文件 {file_path} 失败: {e}")


def mkdir():
    """创建或清空目录"""
    for path in [train_image_path, train_label_path,
                 val_image_path, val_label_path,
                 test_image_path, test_label_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            del_file(path)


def clearfile():
    """清除旧的文件列表"""
    for file_path in [list_train, list_val, list_test]:
        if os.path.exists(file_path):
            os.remove(file_path)


def main():
    mkdir()
    clearfile()

    file_train = open(list_train, 'w')
    file_val = open(list_val, 'w')
    file_test = open(list_test, 'w')

    # 获取所有标注文件
    total_txt = [f for f in os.listdir(label_original_path) if f.endswith('.txt')]
    num_txt = len(total_txt)
    list_all_txt = list(range(num_txt))

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    num_test = num_txt - num_train - num_val

    # 随机划分数据集
    random.shuffle(list_all_txt)
    train = list_all_txt[:num_train]
    val = list_all_txt[num_train:num_train + num_val]
    test = list_all_txt[num_train + num_val:]

    print(f"训练集数目：{len(train)}, 验证集数目：{len(val)}, 测试集数目：{len(test)}")

    # 处理每个文件
    for i, idx in enumerate(list_all_txt):
        txt_name = total_txt[idx]
        name = os.path.splitext(txt_name)[0]  # 去掉扩展名

        # 使用os.path.join构建路径
        srcImage = os.path.join(image_original_path, f"{name}.jpg")
        srcLabel = os.path.join(label_original_path, txt_name)

        # 检查文件是否存在
        if not os.path.exists(srcImage):
            print(f"警告: 图片文件不存在 {srcImage}")
            continue
        if not os.path.exists(srcLabel):
            print(f"警告: 标注文件不存在 {srcLabel}")
            continue

        if idx in train:
            dst_image = os.path.join(train_image_path, f"{name}.jpg")
            dst_label = os.path.join(train_label_path, txt_name)
            shutil.copy2(srcImage, dst_image)
            shutil.copy2(srcLabel, dst_label)
            file_train.write(dst_image + '\n')
        elif idx in val:
            dst_image = os.path.join(val_image_path, f"{name}.jpg")
            dst_label = os.path.join(val_label_path, txt_name)
            shutil.copy2(srcImage, dst_image)
            shutil.copy2(srcLabel, dst_label)
            file_val.write(dst_image + '\n')
        else:  # 测试集
            dst_image = os.path.join(test_image_path, f"{name}.jpg")
            dst_label = os.path.join(test_label_path, txt_name)
            shutil.copy2(srcImage, dst_image)
            shutil.copy2(srcLabel, dst_label)
            file_test.write(dst_image + '\n')

    file_train.close()
    file_val.close()
    file_test.close()
    print("数据集划分完成!")


if __name__ == "__main__":
    main()