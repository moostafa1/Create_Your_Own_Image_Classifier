import os
import torch
import argparse
import numpy as np
from PIL import Image
from random import choice
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from train import DataProcessing, Classifier, check_command_line_arguments




def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments using add_argument() from ArguementParser method
    # 'flowers/test/53/image_03717.jpg'
    parser.add_argument('--img', type = str, default = Prediction().random_image('flowers/test'), help = 'testing image path')
    parser.add_argument('--top_k', type = int, default = 3, help = 'top suggested classes')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'saved model to load')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'saved model to load')
    parser.add_argument('--gpu', type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = 'device use to train the model on')
    return parser.parse_args()



class Prediction:
    def __init__(self):
        pass

    def random_image(self, img_dir):
        imgs_lst = [os.path.join(img_dir, cls, img) for cls in os.listdir(img_dir) for img in os.listdir(os.path.join(img_dir, cls))]
        random_img_path = choice(imgs_lst)
        return random_img_path

    def process_image(self, image):
        # TODO: Process a PIL image for use in a PyTorch model
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        with Image.open(image) as im:
            # resize the image
            img = im.resize((256, 256))
            # center cropping the image
            crop_1 = (256 - 224) / 2
            crop_2 = (256 + 224) / 2
            center_crop = img.crop((crop_1, crop_1, crop_2, crop_2))

            # converting image to numpy array
            np_image = np.array(center_crop)   # , dtype='float64'

            # normalize the image
            np_image = np_image / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            norm_img = (np_image - mean) / std

            # transpose color channels
            norm_img = norm_img.transpose((2, 0, 1))

            # convert numpy array to tensor
            tensor_image = torch.from_numpy(norm_img)

        return tensor_image


    def predict(self, image_path, other, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # TODO: Implement the code to predict the class from an image file
        image = self.process_image(image_path).type(torch.FloatTensor)
        image = image.to(device)

        with torch.no_grad():
            other.model.eval()
            # model expects an input with 4 dimensions which correspond to 
            # BxCxHxW = (Batch x Channel x Height x Width).
            # Since you are testing it with only one image, you are missing the Batch (B) dimension.
            # To solve this, you can add this dimension by using unsqueeze
            log_ps = other.model(image.unsqueeze(0))
            ps = torch.exp(log_ps)

            top_p, top_class = ps.topk(topk, dim=1)
            top_p, top_class = top_p.tolist()[0], top_class.tolist()[0]
        cls_label = [k for ind in top_class for k in other.model.class_to_idx if other.model.class_to_idx.get(k) == ind]

        return top_p, cls_label


    def plot_top_class(self, img_path, top_props, class_labels, cat_to_name):
        img = self.process_image(img_path)

        cls_names = [cat_to_name[k] for ind in class_labels for k in cat_to_name if k == ind]

        plt.figure(figsize = (6,10))
        ax = plt.subplot(2, 1, 1)
        DataProcessing().imshow(img, ax=ax, title=cls_names[0]);

        ax = plt.subplot(2, 1, 2)
        ax.barh(range(len(cls_names)), top_props, tick_label=cls_names);
        ax.invert_yaxis()
        plt.savefig('predicted image.png')
        return cls_names




if __name__ == "__main__":
    # receive command line arguments from user
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg
    check_command_line_arguments(in_arg)

    # set the device to use
    device = in_arg.gpu
    # print(f"You are training your model on --- {device} ---\n\n")

    # get random image to test the model
    random_img_path = in_arg.img

    # create the classifier
    classifier = Classifier()

    # load the model
    classifier.load_model(in_arg.checkpoint)

    # make prediction on the image
    top_p, cls_label = Prediction().predict(random_img_path, classifier, in_arg.top_k)
    # print(top_p)
    # print(cls_label)

    # define category_names dict
    cat_to_name = Classifier().load_json_data(in_arg.category_names)

    # plot the predicted image with its name ant the top suggested classes
    cls_names = Prediction().plot_top_class(random_img_path, top_p, cls_label, cat_to_name)

    # print nearest predictions
    print("predicted image: precentage of prediction")
    print(dict(zip(cls_names, top_p)))