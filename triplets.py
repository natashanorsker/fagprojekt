
from data_loader import *
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#https://github.com/USCDataScience/Image-Similarity-Deep-Ranking/blob/master/triplet_sampler.py
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    '''
    Returns path of folder/directory of every image present in directory by using simple regex expression
    '''
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]



def category_from_id(id):
    #should return the category in a string
    pass
    category = ''

    return category

def show_triplet_pair(list_of_triplet_paths, save=False):

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=1, ncols=3)

    for i, axi in enumerate(ax.flat):
        img = Image.open(list_of_triplet_paths[i])
        axi.imshow(img, alpha=1)
        # get indices of row/column
        rowid = i // 3
        colid = i % 3

    plt.tight_layout(True)
    plt.show()


def get_positive_image_paths(image_path, all_positive_paths, num_pos_images=1):
    #https://github.com/USCDataScience/Image-Similarity-Deep-Ranking/blob/master/triplet_sampler.py
    '''
    image_path (str): path to the image (eg. 'data\\799367C00\\799367C00_02.jpg')
    all_image_paths (list): list of paths to images of same product as image_path (eg. ['data\\799367C00\\799367C00_02.jpg', 'data\\799367C00\\799367C00_03.jpg']..)
    num_pos_images (int): number of positive imagepaths to return (returns 1 if unspecified)
    '''

    random_numbers = np.arange(len(all_positive_paths))
    np.random.shuffle(random_numbers)

    if int(num_pos_images) > (len(all_positive_paths) - 1):
        num_pos_images = len(all_positive_paths) - 1

    pos_count = 0
    positive_images = []

    for random_number in list(random_numbers):
        if all_positive_paths[random_number] != image_path:
            positive_images.append(all_positive_paths[random_number])
            pos_count += 1
            if int(pos_count) > (int(num_pos_images) - 1):
                break
    return positive_images


def get_negative_images(all_image_paths, positive_image_paths, num_neg_images=1):
    '''
    # https://github.com/USCDataScience/Image-Similarity-Deep-Ranking/blob/master/triplet_sampler.py
    '''

    random_numbers = np.arange(len(all_image_paths))
    np.random.shuffle(random_numbers)

    if int(num_neg_images) > (len(all_image_paths) - 1):
        num_neg_images = len(all_image_paths) - 1

    neg_count = 0
    negative_images = []
    for random_number in list(random_numbers):
        if all_image_paths[random_number] not in positive_image_paths:
            negative_images.append(all_image_paths[random_number])
            neg_count += 1
            if neg_count > (int(num_neg_images) - 1):
                break
    return negative_images

def triplet_sampler(num_positive_imgs=1, num_negative_imgs=1, hard_negative_percentage = 0.5):
    '''
    Writes different combination of query, positive + negative images to a csvfile (writes the paths to the images)
    '''

    #get ids for the different classes [ring, earring, etc.]
    classes = sort_by_category(catalog)

    all_img_paths = []

    for id in catalog.keys():
        all_img_paths += list_pictures('data\\'+ id)


    triplets = []
    for class_ in tqdm(classes.keys()):
        semi_negative_ids = classes[class_]
        semi_negative_img_paths = []
        for sem_neg_id in semi_negative_ids:
            semi_negative_img_paths += list_pictures('data\\'+ sem_neg_id)

        for product_id in semi_negative_ids:
            #for all the images of the same product
            image_names = list_pictures('data\\'+product_id)
            for image_name in image_names:
                query_image = image_name
                positive_images = get_positive_image_paths(query_image, image_names, num_positive_imgs)
                for positive_image in positive_images:
                    #find if it should be hard or semi negative:
                    if random.random() > hard_negative_percentage:
                        #semi negative:
                        negative_images = get_negative_images(all_img_paths, image_names, num_negative_imgs)
                    else:
                        #hard negative:
                        negative_images = get_negative_images(all_img_paths, semi_negative_img_paths, num_negative_imgs)
                    for negative_image in negative_images:
                        triplets.append(query_image + ',')
                        triplets.append(positive_image + ',')
                        triplets.append(negative_image + '\n')

    #write to file
    f = open("triplet_pairings.txt", 'w')
    f.write("".join(triplets))
    f.close()





if __name__ == "__main__":
    catalog = dict_from_json("catalog.json")
    triplet_sampler()
