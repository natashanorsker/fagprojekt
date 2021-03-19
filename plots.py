from triplets import list_pictures
from data_loader import*
import seaborn as sns
import matplotlib.pyplot as plt

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