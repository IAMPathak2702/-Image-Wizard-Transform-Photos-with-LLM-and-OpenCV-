import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    """
    Computes the total loss for the neural style transfer process, which includes content loss, style loss, and total variation loss.

    Parameters:
        neural_net (torch.nn.Module): The neural network used for feature extraction.
        optimizing_img (torch.Tensor): The image being optimized.
        target_representations (list): List containing target content and style representations.
        content_feature_maps_index (int): Index of the layer used for content representation.
        style_feature_maps_indices (list): Indices of the layers used for style representation.
        config (dict): Configuration dictionary containing weights for different loss components.

    Returns:
        total_loss (torch.Tensor): The combined loss.
        content_loss (torch.Tensor): The content loss.
        style_loss (torch.Tensor): The style loss.
        tv_loss (torch.Tensor): The total variation loss.
    """
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    # Get the current feature maps from the neural network
    current_set_of_feature_maps = neural_net(optimizing_img)

    # Calculate content loss
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    # Calculate style loss
    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    # Calculate total variation loss
    tv_loss = utils.total_variation(optimizing_img)

    # Combine all losses
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    """
    Creates a function that performs a single optimization step.

    Parameters:
        neural_net (torch.nn.Module): The neural network used for feature extraction.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the image.
        target_representations (list): List containing target content and style representations.
        content_feature_maps_index (int): Index of the layer used for content representation.
        style_feature_maps_indices (list): Indices of the layers used for style representation.
        config (dict): Configuration dictionary containing weights for different loss components.

    Returns:
        tuning_step (function): A function that performs a single optimization step.
    """
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step


def neural_style_transfer(config):
    """
    Performs neural style transfer using the given configuration.

    Parameters:
        config (dict): Configuration dictionary containing all necessary parameters for the NST process.

    Returns:
        dump_path (str): Path to the directory where the output images are saved.
    """
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.splitext(os.path.basename(content_img_path))[0] + '_' + os.path.splitext(os.path.basename(style_img_path))[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)

    if config['init_method'] == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path


if __name__ == "__main__":
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # Saves images in the format: %04d.jpg

    # Parsing the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="Content image name", default='figures.jpg')
    parser.add_argument("--style_img_name", type=str, help="Style image name", default='vg_starry_night.jpg')
    parser.add_argument("--height", type=int, help="Height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="Weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="Weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="Weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="Saving frequency for intermediate images (-1 means only final)", default=-1)
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # Perform neural style transfer
    results_path = neural_style_transfer(optimization_config)
