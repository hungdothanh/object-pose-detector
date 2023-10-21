import yaml
import json
from pathlib import Path
import os

def create_data_yaml(args):
    input_json_path = Path(args.input_json)
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    save_dir = Path(args.save_dir)

    # Read annotation file
    with open(input_json_path, 'r') as f:
        coco = json.load(f)
    
    # Extract class names from the annotation file
    category_names = [category['name'] for category in coco['categories']]

    # Prepare the data dictionary
    data = {
        'train': str(train_path),
        'val': str(val_path),
        'nc': len(category_names),
        'names': category_names
    }

    # Save data to data.yaml under a given directory path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'data.yaml')

    # Write the data dictionary to data.yaml
    with open(save_path, 'w') as f:
        yaml.dump(data, f)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read and create data.yaml file:")
    
    parser.add_argument("-i", "--input_json", dest="input_json",
        help="path to COCO json")
    parser.add_argument("-t", "--train_path", dest="train_path",
        help="path to the folder containing train images")
    parser.add_argument("-v", "--val_path", dest="val_path",
        help="path to the folder containing val images")
    parser.add_argument("-s", "--save_dir", dest="save_dir",
        help="path to save the output data.yaml")

    args = parser.parse_args()

    create_data_yaml(args)


