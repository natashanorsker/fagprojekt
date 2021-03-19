from data_loader import*
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import os
import math


#https://github.com/USCDataScience/Image-Similarity-Deep-Ranking/blob/master/triplet_sampler.py
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    '''
    Returns path of folder/directory of every image present in directory by using simple regex expression
    '''
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]

def dict_from_json(path="catalog.json"):
    # open the product catalog:
    a_file = open(path, "r")
    catalog = json.loads(a_file.read())
    a_file.close()
    return catalog


def category_from_id(id):
    #should return the category in a string
    pass
    category = ''

    return category

def sort_by_category(catalog):
    categories = {'ring': [], 'necklace': [], 'charm': [], 'earring': [], 'bracelet': [], 'misc': []}
    categories_list = list(categories.keys())
    stem_categories = [ps.stem(token) for token in categories_list]

    for id in catalog.keys():
        found = False
        word_list = catalog[id]['product_name'].lower().split(' ')
        stemmed_words = [ps.stem(token) for token in word_list]


        for i in range(len(stem_categories)):
            if stem_categories[i] in stemmed_words:
                categories[categories_list[i]].append(id)
                found=True
                break

            elif 'bangl' in stemmed_words:
                categories['bracelet'].append(id)
                found=True
                break

            elif 'pendant' in stemmed_words:
                categories['necklace'].append(id)
                found=True
                break

        if not found:
            categories['misc'].append(id)
    return categories



def show_images(list_of_image_paths, ncols, plot_title=True, save=False):
    plt.box(False)
    n_imgs = len(list_of_image_paths)
    nrows = math.ceil(n_imgs/ncols)

    try:
        list_of_image_paths[ncols]
    except IndexError:
        print('Error: ncols > len(images). There should be less columns than the amount of total images.')
        return

    if n_imgs == 1:
        img = Image.open(list_of_image_paths[0])
        plt.imshow(img)
        plt.axis('off')
        plt.title(list_of_image_paths[0].split('\\')[-1][:-4])

    else:

        # create figure (fig), and array of axes (ax)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)


        for i, axi in enumerate(ax.flat):
            if i < n_imgs:
                img = Image.open(list_of_image_paths[i])
                title = list_of_image_paths[i].split('\\')[-1][:-4]
                axi.imshow(img, alpha=1)
                axi.axis('off')
                axi.set_title(title)

    if save:
        plt.save('plotted_imgs.png')

    plt.show()


def occurrence_plot(catalog=catalog):
    occurrences = np.zeros(42)

    for id in catalog.keys():
        images = list_pictures('data\\' + id)
        images_no_au = [img for img in images if 'AU' not in img]
        occurrences[len(images_no_au)] += 1

    sns.set_style('whitegrid')
    sns.barplot(x=list(range(42)), y=occurrences)
    plt.xlabel('Number of images')
    plt.ylabel('Number of products')
    plt.xticks(list(range(42))[::2])
    plt.show()