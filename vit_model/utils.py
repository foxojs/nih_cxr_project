def extract_patches(image_tensor, patch_size = 4):
    bs, c, h, w = image_tensor.size()

    # define teh unfold layer with appropriate parameters 

    unfold = torch.nn.Unfold(kernel_size = patch_size, stride = patch_size)

    unfolded = unfold(image_tensor)

    # reshape the unfolded tensor to match the desired output shape 
    # output shaep BS x L x C x 8 x8 where L is the number of patches in each dimension 
    # fo reach dimension, number of patches = (original dimension size) //patch_size 

    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded
