from PIL import Image
import os

print(os.listdir('/'))

def resize_images(input_dir, output_dir, size=(128, 128)):
    print("Input directory:", input_dir)
    print("Directories inside input directory:", os.listdir(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue
        save_path = os.path.join(output_dir, category)
        os.makedirs(save_path, exist_ok=True)

        for subcategory in os.listdir(category_path):
            subcategory_path = os.path.join(category_path, subcategory)
            if not os.path.isdir(subcategory_path):
                continue
            sub_save_path = os.path.join(save_path, subcategory)
            os.makedirs(sub_save_path, exist_ok=True)

            for img_file in os.listdir(subcategory_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    continue
                img_path = os.path.join(subcategory_path, img_file)
                img = Image.open(img_path).resize(size)
                img.save(os.path.join(sub_save_path, img_file))

if __name__ == "__main__":
    resize_images('/Users/dipaktomar/Documents/vegetable_fruits_dataset', 'resized_dataset', size=(128, 128))
