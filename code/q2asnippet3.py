num_epochs = 500
learning_rate = 1e-1
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(epoch)
    eta = Variable(eta)
    # ===================forward=====================
    output = net(eta)
    loss = criterion(output, noisy_img)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # ===================log========================
    #print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))

# Shows final result
out = net(eta)
out_img = out[0, 0, :, :].transpose(0,1).detach().numpy()
