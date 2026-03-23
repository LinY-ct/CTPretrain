import glob
import os
import random

list1 = [1000000, 500000, 100000, 50000, 10000]
list2 = [240, 180, 144, 120, 90, 80, 72, 60, 48, 36, 18]
list3 = [360, 240, 180, 120]
# 使用两层循环生成两两配对

pairs = [('ld_sv' ,x, y) for x in list1 for y in list2]
pairs += [('ld_lv', x, y) for x in list1 for y in list3]
pairs += [('ld', x, 720) for x in list1 ]
pairs += [('sv', 1000000, x,) for x in list2 ]
pairs += [('lv', 1000000,x) for x in list3 ]
print("pairs num: " , len(pairs))


paths_list = []
# for i in all_dir:
paths_list += glob.glob(os.path.join('/mnt/nas/wsy/MayoData/npy/', "L067") + '/*/*', recursive=True)


random.shuffle(paths_list)
# paths_list = paths_list[:5000]


selected_numbers = random.choices(pairs, k=len(paths_list))
random.shuffle(selected_numbers)

paths_list = list(zip(paths_list, selected_numbers))


with open("test.txt", "w") as file:  # 'w' 表示写入模式
    for item in paths_list:
        
        file.write(item[0] + ',')
        for i in item[1]:
            file.write(str(i) + ',')
        file.write('\n')
print("数据已写入文件")