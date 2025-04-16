from torchvision import transforms

transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
])


transform_val = transforms.Compose([
            transforms.ToTensor(),
])
