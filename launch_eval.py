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
    dataset_path = "./dataset/MVTec"
    class_name   = "SimplePatchCore"
    weights_path = "./experiments/weights-Patchcore-mobilenet"

    all_categories = utils_list_categories(dataset_path)[7:]

    for category in tqdm(all_categories):
        print("[INFO] Category {}".format(category))
        cmd = """python evaluate.py --module_path models.patchcore \
                                    --class_name SimplePatchcore \
                                    --weights_path {} \
                                    --category {} \
                                    --dataset_path {}
                                    """.format(weights_path, category, dataset_path)
        os.system(cmd)