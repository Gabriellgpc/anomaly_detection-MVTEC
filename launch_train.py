from tqdm import tqdm
import os

from uuid import uuid4

def utils_list_categories(root_dir):
    import os
    list_dir = os.listdir(root_dir)
    categories = []
    for path in tqdm(list_dir):
        if os.path.isdir( os.path.join(root_dir, path) ):
            categories.append(path)
    categories = sorted(categories)
    return categories

if __name__ == "__main__":
    data_root = "datasets/MVTecAD"
    all_categories = utils_list_categories(data_root)

    exp_id = uuid4().hex[:4]
    for category in all_categories:
        cmd = "python train.py --category {} --exp_id {}".format(category, exp_id)
        os.system(cmd)