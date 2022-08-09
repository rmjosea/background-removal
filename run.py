from rembg import remove
from PIL import Image
import numpy as np
import cv2
import os
from skimage import io

# Get The Current Directory
currentDir = os.path.dirname(__file__)

inputs_dir = os.path.join(currentDir, 'images/')
results_dir = os.path.join(currentDir, 'images')
masks_dir = os.path.join(currentDir, 'images')
input_path = 'images/288_2_front.jpg'
output_path = 'images/288_front_out.png'

def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = np.frombuffer(predict, np.uint8)
    #predict = predict.squeeze()
    #predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict*255).convert('RGB')
    image = io.imread(image_name)
    #imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(image)

    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        im = np.concatenate((image, mask), axis=2)
        im = Image.fromarray(im, 'RGBA')

    im.save(d_dir+output_name)

def bg_as_bytes(input_path,output_path):

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input, post_process_mask=True, alpha_matting=True,alpha_matting_erode_size=20)

            save_output(inputs_dir + '288_2_front' + '.jpg', 'test_file_name' +
                        '.png', output, 'images/', 'image')
            save_output(inputs_dir + '288_2_front' + '.jpg', 'test_file_name_mask' +
                        '.png', output, 'images/', 'mask')
            o.write(output)

def bg_as_PIL(input_path, output_path):

    input = Image.open(input_path)
    output = remove(input, post_process_mask=True, alpha_matting=True)

    output.save(output_path)

def bg_as_np(input_path, output_path):
    input = cv2.imread(input_path)
    output = remove(input, post_process_mask=True, alpha_matting=True)
    cv2.imwrite(output_path, output)


bg_as_bytes(input_path,output_path)
#bg_as_PIL(input_path,output_path)
#bg_as_np(input_path,output_path)
