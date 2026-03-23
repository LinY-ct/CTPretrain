import glob
import os
import random

list1 = [1000000, 500000, 100000, 50000, 10000]
list2 = [240, 180, 144, 120, 90, 80, 72, 60, 48, 36, 18]
list3 = [360, 240, 180, 120]
# 使用两层循环生成两两配对

pairs = [('ld_sv' ,x, y) for x in list1 for y in list2]
pairs += [('ld_lv', x, y) for x in list1 for y in list3]
pairs += [('ld', x) for x in list1 ]
pairs += [('sv', x) for x in list2 ]
pairs += [('lv', x) for x in list3 ]
print("pairs num: " , len(pairs))

all_dir = os.listdir('/mnt/nas/wsy/DeepLesion/Image_png/')
dirs = all_dir[-45:]
paths_list = []
for i in dirs:
    paths_list += glob.glob(os.path.join('/mnt/nas/wsy/DeepLesion/Image_png/', i) + '/*.png', recursive=True)
random.shuffle(paths_list)
paths_list = paths_list[:500]


selected_numbers = random.choices(pairs, k=len(paths_list))
random.shuffle(selected_numbers)

paths_list = list(zip(paths_list, selected_numbers))
print("paths_list num:" , len(paths_list))

with open("deeplession_test.txt", "w") as file:  # 'w' 表示写入模式
    for item in paths_list:

        file.write(item[0] + ',')
        for i in item[1]:
            file.write(str(i) + ',')
        file.write('\n')
print("数据已写入文件")


