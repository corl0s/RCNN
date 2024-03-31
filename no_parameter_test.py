def calculate_parameters(config):
    total_params = 0
    input_channels = 3  # Assuming input images have 3 channels (e.g., RGB)

    for layer in config:
        params = 0  # Initialize params for each layer
        if layer == "M":
            # Max-pooling layer has no parameters
            continue
        elif isinstance(layer, tuple):  # Convolutional layer
            kernel_size, filters, _, _ = layer
            params = (input_channels * filters * kernel_size**2) + filters
            input_channels = filters  # Update input channels for the next layer
        elif isinstance(layer, list):  # Sequence of convolutional layers
            for sub_layer in layer:
                params += calculate_parameters([sub_layer])
        total_params += params

    return total_params


architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# Assuming the input image has 3 channels
total_parameters = calculate_parameters(architecture_config)
print(f"Total number of parameters in the model: {total_parameters}")