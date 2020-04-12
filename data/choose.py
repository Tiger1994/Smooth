import os
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':
    input_path = ''
    output_path = ''
    save_path = '/SSD64/Smooth/train/class5'
    choose_class = '5'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for name in os.listdir(input_path):
        name = name.split('.')[0]
        choose_path = output_path+'/'+name+'_'+choose_class+'.png'
        choose_image = Image.open(choose_path)
        save_name = save_path+'/'+name+'.png'
        choose_image.save(save_name)
