import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import data_loader

if __name__ == '__main__':
    # data_path = '/dataset/img_align_celeba'
    # attr_path = './list_attr_celeba.txt'
    data_path = '/content/celeba/img_align_celeba/img_align_celeba'
    attr_path = '/acgan/list_attr_celeba.txt'
    image_size = 128
    batch_size = 64
    workers = 0
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # img_path = '/dataset/img_align_celeba'
    # attr_path = './list_attr_celeba.txt'

    img_path = '/content/celeba/img_align_celeba/img_align_celeba'
    attr_path = '/acgan/list_attr_celeba.txt'

    dataset = data_loader.CelebA_Slim(img_path=img_path,
                                      attr_path=attr_path,
                                      transform=transform,
                                      slice=[0, -1],
                                      attr=[8, 9, 11, 15, 20, 25, 31, 39])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    print(dataset.get_classes())
    print(len(dataset))
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Training Images")
    print(real_batch[1][:8])
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:8], nrow=8, padding=0, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
