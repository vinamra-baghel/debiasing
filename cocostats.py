from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from scipy.stats import spearmanr
import seaborn as sns


# Parse arguments
parser = argparse.ArgumentParser(description='COCO 2014 statistics')
parser.add_argument('--annotations_file_path', type=str, default='/raid/ganesh/prateekch/debiasing/MSCOCO_2014',
                    help='path to the annotations file for the MSCOCO 2014 dataset')
parser.add_argument('--output_dir', type=str, default='/raid/ganesh/prateekch/debiasing/plots/',
                    help='path to the output directory where the statistics will be saved')
parser.add_argument('--type', type=str, default='objp/distribution', help='type of training objective')


args = parser.parse_args()

class coco2014:
    def __init__(self, annotations_file_path):
        self.annotations_file_path = annotations_file_path
        self.coco = COCO(self.annotations_file_path)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = [category['id'] for category in self.categories]
        print(self.category_ids)
        self.image_ids = self.coco.getImgIds()
        self.annotations = self.coco.loadAnns(self.coco.getAnnIds())

        self.total_images, self.total_annotations, self.average_annotations_per_image = self.print_statistics(return_stats=True)

        self.cat2cat = dict()                       ## to map category ids to integer indices, only for internal usage
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)


        self.class_frequencies = {self.cat2cat[category_id]: 0 for category_id in self.category_ids}
        self.get_class_frequencies()

    def get_class_frequencies(self):
        for img_id in self.image_ids:
            s = set()
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                s.add(self.cat2cat[ann['category_id']])
            for i in s:
                self.class_frequencies[i]+=1

    def print_statistics(self, return_stats=False):
        total_images = len(self.image_ids)
        total_annotations = len(self.annotations)
        average_annotations_per_image = total_annotations / total_images
        if not return_stats:
            print("Total images:", total_images)
            print("Total annotations:", total_annotations)
            print("Average annotations per image:", average_annotations_per_image)
        else:
            return total_images, total_annotations, average_annotations_per_image

        # print(self.cat2cat)
        # self.print_class_frequencies()

    # def print_class_frequencies_indexed(self):
    #     for category_id, frequency in self.class_freq_indexed.items():
    #         print(category_id, frequency)

    def print_class_frequencies(self):
        for category_id, frequency in self.class_frequencies.items():
            print(category_id, frequency)

def getweight():
    a = np.array([1.3506e-02, 2.2401e-04, 3.7973e-03, 3.5738e-04, 9.8529e-05, 2.2142e-04,
        2.6827e-05, 1.3337e-04, 5.2513e-04, 7.3525e-05, 4.3959e-05, 2.2246e-04,
        2.2749e-05, 5.6700e-04, 4.8376e-04, 4.8225e-04, 8.1473e-04, 9.1966e-04,
        7.7134e-04, 6.6144e-04, 9.4522e-06, 1.0691e-04, 1.0266e-04, 1.6579e-04,
        9.0846e-04, 1.1390e-04, 4.6872e-03, 5.4677e-05, 5.5054e-05, 2.1692e-04,
        1.4689e-04, 3.8521e-05, 3.9471e-03, 5.1816e-05, 5.6270e-04, 2.0002e-02,
        4.7161e-05, 4.5167e-04, 1.2655e-03, 5.3536e-03, 8.8023e-02, 5.5391e-01,
        9.5253e-01, 7.8818e-01, 9.9817e-01, 9.7028e-01, 5.5160e-05, 1.8901e-04,
        4.5830e-04, 2.5398e-04, 5.7551e-05, 1.1605e-04, 1.6816e-04, 2.6326e-04,
        5.4254e-05, 4.1939e-04, 8.6797e-02, 2.8652e-03, 7.0724e-02, 3.3828e-04,
        9.1159e-01, 2.1481e-04, 6.3819e-02, 2.0327e-03, 4.5793e-03, 9.8279e-04,
        1.0653e-02, 3.2280e-03, 2.8006e-02, 7.3537e-03, 3.5776e-05, 1.9266e-04,
        3.1540e-01, 5.2232e-02, 9.9780e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        5.4036e-03, 1.1308e-02])

    return a

def plot(weightfunc,outputdir, type):   ## weightfunc is the weight function to be used
    # weight = getweight()
    weight = weightfunc()

    trainpath = os.path.join(outputdir,type,'train')
    valpath = os.path.join(outputdir,type,'val')

    # train class frequencies
    x = np.array([i for i in range(80)])

    indexes = np.array(list(coco2014_train.class_frequencies.values())).argsort()

    # print(indexes)

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Class Frequencies and Weights')
    axs[0].bar(x,np.array(list(coco2014_train.class_frequencies.values()))[indexes])
    axs[0].set_title('Train Class Frequencies')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Train Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    # plt.savefig('class_freq_weight_train_sort.png')
    plt.savefig(os.path.join(trainpath,'class_freq_weight_train_sort.png'))

    ## Manual weights assigned according to inverse frequencies
    invFreqWeights = 1/np.array(list(coco2014_train.class_frequencies.values()))

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Class Frequencies and Weights')
    axs[0].bar(x,invFreqWeights[indexes])
    axs[0].set_title('Train Class Inverse Frequency Weights')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Train Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    plt.savefig(os.path.join(trainpath,'class_inv_freq_weight_train_sort_sum.png'))

    ## val class frequencies
    indexes = np.array(list(coco2014_val.class_frequencies.values())).argsort() 
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    fig.suptitle('Class Frequencies and Weights')
    # axs[0].bar(coco2014_val.class_frequencies.keys(), coco2014_val.class_frequencies.values())
    axs[0].bar(x,np.array(list(coco2014_val.class_frequencies.values()))[indexes])
    axs[0].set_title('Val Class Frequencies')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    # axs[1].bar(coco2014_val.class_frequencies.keys(), weight)
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Val Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    # plt.savefig('class_freq_weight_val_sort.png')
    plt.savefig(os.path.join(valpath,'class_freq_weight_val_sort.png'))


    ## plotting with the ratio of positive to negative imbalance
    a = len(coco2014_train.image_ids) - np.array(list(coco2014_train.class_frequencies.values()))
    b = np.array(list(coco2014_train.class_frequencies.values()))
    print(a)
    print(b)

    arr = b/a
    print(arr)
    inde = arr.argsort()
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    print(inde)
    print(arr[inde])

    fig.suptitle('Class Frequencies and Weights')
    # axs[0].bar(coco2014_train.class_frequencies.keys(), b/a)
    axs[0].bar(x,b[indexes]/a[indexes])
    axs[0].set_title('Train Class Imbalance ratio')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    # axs[1].bar(coco2014_train.class_frequencies.keys(), weight)
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Train Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')

    # plt.savefig('class_freq_weight_train_ratio_sort.png')
    plt.savefig(os.path.join(trainpath,'class_freq_weight_train_ratio_sort.png'))

    ## val class frequencies
    a = len(coco2014_val.image_ids) - np.array(list(coco2014_val.class_frequencies.values()))
    b = np.array(list(coco2014_val.class_frequencies
                .values()))

    indexes = (b/a).argsort()

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle('Class Frequencies and Weights')
    # axs[0].bar(coco2014_val.class_frequencies
                # .keys(), b/a)
    axs[0].bar(x,b[indexes]/a[indexes])
    axs[0].set_title('Val Class Imbalance ratio')
    axs[0].set_xlabel('Class Index')
    axs[0].set_ylabel('Frequency')
    # axs[1].bar(coco2014_val.class_frequencies.keys(), weight)
    axs[1].bar(x,weight[indexes])
    axs[1].set_title('Val Class Weights')
    axs[1].set_xlabel('Class Index')
    axs[1].set_ylabel('Weight')


    # plt.savefig('class_freq_weight_val_ratio_sort.png')
    plt.savefig(os.path.join(valpath,'class_freq_weight_val_ratio_sort.png'))

def rankcorrelation(cocotrain, cocoval):
    training_freq = list(cocotrain.values())
    validation_freq = list(cocoval.values())

    # Rank the frequencies
    training_rank = [sorted(training_freq).index(x) + 1 for x in training_freq]
    validation_rank = [sorted(validation_freq).index(x) + 1 for x in validation_freq]

    # print(training_rank)
    # print(validation_rank)

    # Compute the rank correlation coefficient
    correlation_coefficient, p_value = spearmanr(training_rank, validation_rank)

    return correlation_coefficient, p_value


def plotfreqBar(cocotrain, cocoval, savepath='/raid/ganesh/prateekch/debiasing/'):
    ## plot a bar plot of the frequencies of the classes with train and val side by side for each class
    classes = list(cocotrain.class_frequencies.keys())
    training_freq = np.array(list(cocotrain.class_frequencies.values()))
    validation_freq = np.array(list(cocoval.class_frequencies.values()))

    sns.set_theme()
    # Set the width of the bars
    bar_width = 0.3

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(classes))
    # r2 = [x + bar_width for x in r1]


    # Create the bar plot
    plt.figure(figsize=(15,9))
    plt.bar(r1, training_freq, color='b', width=bar_width, label='Train')
    plt.bar(r1+bar_width, validation_freq, color='r', width=bar_width, label='Validation')

    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Frequencies')
    plt.xticks([r + bar_width/2 for r in range(len(classes))], classes)
    plt.title('Train vs Validation Frequencies for Each Class')

    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(savepath, 'train_val_freq.png'))

    ## plotting it by normalizing the frequencies
    plt.close()

    plt.figure(figsize=(15,9))
    plt.bar(r1, training_freq/cocotrain.total_images, color='b', width=bar_width, label='Train',align='center')
    plt.bar(r1+bar_width, validation_freq/cocoval.total_images, color='r', width=bar_width, label='Validation',align='center')

    # Add labels and title
    plt.xlabel('Classes' )
    plt.ylabel('Frequencies' )
    plt.xticks([r + 0.1 for r in range(len(classes))], classes)
    plt.title('Train vs Validation fraction of positive samples for Each Class')

    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(savepath, 'train_val_normalised.png'))

    plt.close()
    ## same above but without class 0
    classes = classes[1:]
    training_freq = training_freq[1:]
    validation_freq = validation_freq[1:]

    plt.figure(figsize=(15,9))
    plt.bar(r1[1:], training_freq/cocotrain.total_images, color='b', width=bar_width, label='Train',align='center')
    plt.bar(r1[1:]+bar_width, validation_freq/cocoval.total_images, color='r', width=bar_width, label='Validation',align='center')
    plt.xlabel('Classes')
    plt.ylabel('Frequencies')
    plt.xticks([r + 0.1 for r in range(len(classes))], classes)
    plt.title('Train vs Validation fraction of positive samples for Each Class (excluding class 0)')
    plt.legend()
    plt.savefig(os.path.join(savepath, 'train_val_normalised_no0.png'))


if __name__ == '__main__':
    
    train_annotation_file_path = os.path.join(args.annotations_file_path, 'annotations/instances_train2014.json')
    val_annotation_file_path = os.path.join(args.annotations_file_path, 'annotations/instances_val2014.json')


    coco2014_train = coco2014(train_annotation_file_path)
    coco2014_val = coco2014(val_annotation_file_path)

    # print("---------------Train Statistics---------------")
    # coco2014_train.print_statistics()
    # coco2014_train.print_class_frequencies()

    # print("---------------Val Statistics---------------")
    # coco2014_val.print_statistics()
    # coco2014_val.print_class_frequencies()


    print("---------------Rank Correlation---------------")
    correlation_coefficient, p_value = rankcorrelation(coco2014_train.class_frequencies, coco2014_val.class_frequencies)
    print("Rank correlation coefficient:", correlation_coefficient)
    print("P-value:", p_value)

    """

        Rank correlation coefficient: 0.990693749408758
        P-value: 2.522595515473387e-69

    """

    print("---------------Plotting Frequencies---------------")
    plotfreqBar(coco2014_train, coco2014_val)
    # plot(getweight,args.output_dir, args.type)

