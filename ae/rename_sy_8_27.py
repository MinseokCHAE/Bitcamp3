import os
import shutil

input_dir = r'C:\Users\user\Desktop\test'
input_dir = input_dir.replace('\\','/')
print(input_dir)
print(os.path.isdir(input_dir))

save_dir = input_dir+'_c'
print(save_dir)

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

try:
    # 경로 유무 확인
    if not os.path.isdir(input_dir):
        raise FileNotFoundError

    # 폴더에 다른 사진이나 자료가 있을 시 에러를 발생시키고 종료. 
    # 사진들이 덮어 씌워짐 방지.     
    if os.listdir(save_dir):
        raise SystemExit

except FileNotFoundError:
    print('Could not find the directory!')
    exit()

except SystemExit:
    print('Error: ', save_dir, 'is not empty.')
    exit()

# 폴더명
folder_names = os.listdir(input_dir)

for folder_name in folder_names:
    sub_dir = os.path.join(input_dir,folder_name)
    # 파일명
    file_names = os.listdir(sub_dir)
    for file_name in file_names:
        # print('sub_dir:',sub_dir)
        rename = (folder_name+'_'+file_name) # 폴더명_파일명으로 rename
        # print('2222222222',rename)
        print('3333333',os.path.join(save_dir, rename))
        shutil.move(os.path.join(sub_dir, file_name), os.path.join(save_dir, rename))

