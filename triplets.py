
from data_loader import *

#https://github.com/USCDataScience/Image-Similarity-Deep-Ranking/blob/master/triplet_sampler.py
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    '''
    Returns path of folder/directory of every image present in directory by using simple regex expression
    '''
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]

'''
def sort_by_category(catalog):
    categories = {'ring': [], 'necklace': [], 'charm': [], 'earring': [], 'bracelet': [], 'misc': []}
    # should return a dict with a list of ids for each category(key) in the dict
    pass

    return categories
'''



def category_from_id(id):
    #should return the category in a string
    pass
    cat = ''

    return cat


def get_positive_image_paths(image_path, product_id, num_pos_images):
    image_dir = 'data\\'+str(product_id)
    #get a list of all imagepaths
    image_names = list_pictures(image_dir)

    #get random image from list of images
    random_numbers = np.arange(len(image_names))
    np.random.shuffle(random_numbers)

    if int(num_pos_images)>(len(image_names)-1):
        num_pos_images = len(image_names)-1

    pos_count = 0
    positive_images = []

    for random_number in list(random_numbers):
        if image_names[random_number] != image_path:
            positive_images.append(image_names[random_number])
            pos_count += 1
            if int(pos_count) > (int(num_pos_images) - 1):
                break
    return positive_images





class TripletDataset:

    def __init__(self):
        pass

    def get_triplets(self):
        pass





if __name__ == "__main__":
    catalog = dict_from_json("catalog.json")
    classes = sort_by_category(catalog)
