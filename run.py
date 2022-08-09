from rembg import remove
import os

# Get The Current Directory
currentDir = os.path.dirname(__file__)

inputs_dir = os.path.join(currentDir, 'images/')
results_dir = os.path.join(currentDir, 'images')
masks_dir = os.path.join(currentDir, 'images')

input_path = 'images/jose_ropa_2.JPG'
output_path = 'images/jose_ropa_2_out.png'
mask_out_path = 'images/jose_ropa_out_mask.png'

def bg_as_bytes(input_path,image_output_path, mask_output_path):
    with open(input_path, 'rb') as i:
        with open(image_output_path, 'wb') as o:
            input = i.read()
            mask_output, img_output = remove(input, post_process_mask=True, alpha_matting=True,alpha_matting_erode_size=30)

            mask_output.save(fp=mask_output_path, format="PNG")
            o.write(img_output)

bg_as_bytes(input_path,output_path,mask_out_path)
