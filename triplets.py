
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


def get_positive_image_paths(image_name, product_id, num_pos_images):
    image_dir = 'data\\'+str(product_id)
    image_names = list_pictures(image_dir)

    return image_names





class TripletDataset:

    def __init__(self):
        pass

    def get_triplets(self):
        pass





if __name__ == "__main__":
    catalog = dict_from_json("catalog.json")
    classes = sort_by_category(catalog)
