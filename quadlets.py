
from data_loader import *
import random
import numpy as np
import csv
from utilities import *



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



def get_images(query_path, possible_image_paths, num_images=1):

    random_numbers = np.arange(len(possible_image_paths))
    np.random.shuffle(random_numbers)

    if int(num_images) > (len(possible_image_paths) - 1):
        num_pos_images = len(possible_image_paths) - 1

    count = 0
    sampled_images = []

    for random_number in list(random_numbers):
        if possible_image_paths[random_number] != query_path:
            sampled_images.append(possible_image_paths[random_number])
            count += 1
            if int(count) > (int(num_images) - 1):
                break
    return sampled_images


def quadlet_sampler(num_positive_imgs=1, num_negative_imgs=1, num_semi_imgs=1):
    '''
    Writes different combination of query, positive + negative images to a csvfile (writes the paths to the images)
    '''

    #get ids for the different classes [ring, earring, etc.]
    catalog = dict_from_json('catalog.json')
    classes = dict_from_json('catalog_by_category.json') #catalog_by_category
    has_no_class = dict_from_json('id_not_in_masterfile.json')

    quadlets = []
    #start with big category instead
    for class_ in tqdm(classes.keys()):
        class_ids = classes[class_]
        negative_ids = list(set(catalog.keys()) - set(class_ids + has_no_class)) #all ids that doesnt belong to this class and don't include the ones that doesn't have a class

        negative_img_paths = []
        for neg_id in negative_ids:
            negative_img_paths += list_pictures('data\\' + neg_id)

        # positive image if it belongs to the same product:
        # semi image if it belongs to the same class, but not the same product:
        for id in class_ids:
            # positive image if it belongs to the same product:
            positive_ids = [id]
            positive_img_paths = list_pictures('data\\' + id) #get all the images of the same product
            
            # semi image if it belongs to the same class, but not the same subclass:
            semi_ids = list(set(class_ids)-set(positive_ids))
            semi_img_paths = [] #get all the images belonging to the semi ids list
            for sem_id in semi_ids:
                semi_img_paths += list_pictures('data\\' + sem_id)

  
            for query_img in positive_img_paths:
                #all positive images to be paired with query image:
                positive_images = get_images(query_img, positive_img_paths, num_positive_imgs)

                #for every positive image, find semi and negatives:
                for pos_image in positive_images:
                    #find a list of all possible semi ids: ie. all images, that are in the same item category, but aren't in the same subcategory:
                    semi_images = get_images(query_img, semi_img_paths, num_semi_imgs)

                    for sem_image in semi_images:
                        negative_images = get_images(query_img, negative_img_paths, num_semi_imgs)

                        for neg_image in negative_images:
                            quadlets.append(query_img + ',')
                            quadlets.append(pos_image + ',')
                            quadlets.append(sem_image + ',')
                            quadlets.append(neg_image + '\n')

    #write to file
    f = open("quadlet_pairings.txt", 'w')
    f.write("".join(quadlets))
    f.close()


if __name__ == "__main__":
    catalog = dict_from_json("catalog.json")
    quadlet_sampler()

    with open('quadlet_pairings.txt', newline='') as csvfile:
        data = list(csv.reader(csvfile))




