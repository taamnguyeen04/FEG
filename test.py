import torch
def domain_loss(d_out, c):
    """

    """
    # # one-hot encoding
    print(c)

    v_out = torch.full_like(d_out, 0)
    v_out[c] = 1
    print(v_out)
    # loss function
    loss = -v_out*torch.log(d_out)
    loss=loss.sum()

    return loss

# rand = torch.randn(4).cuda()
# d_out = torch.exp(rand)/torch.exp(rand).sum()
# print(d_out)
# c = torch.tensor([1, 2]).cuda()
# domain_loss(d_out, c)

# Tensor mẫu c
c = torch.tensor([[0., 0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1., 0.]], device='cuda:0')

# Chuyển đổi các phần tử của c thành số nguyên
c_int = c.long()
print(c)
print("Tensor c sau khi chuyển đổi thành số nguyên:")
print(c_int)