import os
import shutil
src_path = '/data/liucd/data/ZXL/'
dst_path = '/home/zhenghuadong/project/Preprocessing-Pipeline-on-Brain-MR-Images-master/data/anli_for_windows_test/anli_prep_reg_sk_N4_eh/'

for file_path in os.listdir(src_path):
    for file in os.listdir(os.path.join(src_path, file_path)):
        if file.split('.')[-1] == 'gz':
            print(file)
            shutil.copyfile(os.path.join(src_path, file_path, file), os.path.join(dst_path, file_path, file))
