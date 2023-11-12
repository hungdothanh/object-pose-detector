from pycocotools.coco import COCO
import requests
from pathlib import Path
import json
import os

def extract_category_names(json_file_path):
    with open(json_file_path, 'r') as f:
        coco = json.load(f)

    category_names = [category['name'] for category in coco['categories']]
    return category_names

# Convert bbox from xml format (x_min, y_min, width, height) to yolo format (x_center, y_center, width, height)
def xml_to_yolo_bbox(bbox, w, h):
    x_center = (bbox[0] + bbox[2]/2) / w
    y_center = (bbox[1] + bbox[3]/2) / h
    width = bbox[2] / w
    height = bbox[3] / h
    return [x_center, y_center, width, height]

def main(args, download_limit):
    input_json_path = Path(args.input_json)
    folder_path = Path(args.save_dir)

    # initialize COCO api for instance annotations
    coco = COCO(input_json_path)


    #--------Download images to the specified folder----------#
    # Create a folder to store the images
    directory_name = 'images'
    img_folder = os.path.join(folder_path, directory_name)

    # Check if the directory exists, create it if not
    if not os.path.exists(img_folder):
        print("Directory to store downloaded images not found. Creating...")
        os.makedirs(img_folder)

    cats = extract_category_names(input_json_path)
    print(cats)
    downloaded_images = []
    for cat in cats:
        catIds = coco.getCatIds(catNms=cat) 
        imgIds = coco.getImgIds(catIds=catIds)
        images = coco.loadImgs(imgIds)
        print("Number of images of " + str(cat) + " :", len(imgIds))

        img_count = 0
        print("Downloading images...")
        for im in images:
            if im not in downloaded_images:
                img_data = requests.get(im['coco_url']).content

                img_file_path = os.path.join(img_folder, im['file_name'])
                with open(img_file_path, 'wb') as handler:
                    handler.write(img_data)

                downloaded_images.append(im)
                img_count += 1

                # Download full set of images or an inputed amount depending on the user's choice
                if down_var == 'n':
                    if img_count >= int(download_limit):    
                        break
                elif down_var == 'y':
                    continue

        print("Images downloaded:", img_count)


    #-----------Save annotations to new file---------
    # Create a folder to store the labels
    directory_name2 = 'labels'
    val_folder = os.path.join(folder_path, directory_name2)
    
    # Check if the directory exists, create it if not
    if not os.path.exists(val_folder):
        print("Directory to store created labels not found. Creating...")
        os.makedirs(val_folder)

    label_count = 0
    for img in downloaded_images:
        im_width = img['width']
        im_height = img['height'] 
        # load annotations for this image
        imgId = img['id']
        ann_ids = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(ann_ids)
    
        label_file_path = os.path.join(val_folder, img['file_name'].replace('.jpg', '.txt'))
        with open(label_file_path, 'w') as f:
            for ann in anns:
                yolo_bbox = xml_to_yolo_bbox(ann['bbox'], im_width, im_height)
                f.write(str(ann['category_id']-1) + ' ' + ' '.join([str(coord) for coord in yolo_bbox]) + '\n')

        label_count += 1

    print("Labels created:", label_count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load COCO JSON:")
    
    parser.add_argument("-i", "--input_json", dest="input_json",
        help="path to a json file in coco format")
    parser.add_argument("-s", "--save_dir", dest="save_dir",
        help="path to the folder, in which the downloaded images and their labels are saved")
    args = parser.parse_args()

    down_var = input("Do you want to download full set of images? (y/n) ")
    download_limit = None
    if down_var == 'n':
        download_limit = input("How many images for each category do you want to download? ")
    elif down_var == 'y':
        pass
    else:
        print("Invalid input. Please try again.")
        quit()

    main(args,download_limit)







