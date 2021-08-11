'''    
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
#load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    #with torch.no_grad():
    batch.requires_grad_(True)
    prediction, myAs, mySigs = model(Variable(batch, requires_grad = True), CUDA)
    #print(grads)
    #prediction.backward()
    print("\nPrediction")
    print(prediction.shape)
    ders = []
    print("\nA's")
    for tens in myAs:
        print(tens.shape)
    myA = (myAs[0] + myAs[1] +myAs[2])/3
    A = torch.sum(torch.sum(myA))
    print("\nDerivatives")
    for tens in derivatives:
        print(tens.shape)
        ders.append(float(torch.sum(tens)))
    mySigs[0] = torch.mean(mySigs[0], 0)
    print("\nShape of mySigs:")
    print(mySigs[0].shape)
    coeff_1 = 1 - 2 * mySigs[0]
    print("\nShape of coeff_1:")
    print(coeff_1.shape)
    coeff_2 = 6 * mySigs[0]**2 - 6 * mySigs[0] + 1
    print("\nShape of coeff_2:")
    print(coeff_2.shape)
    numerator = coeff_1 * (ders[0]**2)
    print("\nShape of numerator:")
    print(numerator.shape)
    denominator = 2*numerator + (coeff_2 * A * (ders[0]**3))
    print("\nShape of denominator:")
    print(denominator.shape)
    alpha = numerator/denominator
    print("\nShape of alpha:")
    print(alpha.shape)

    print("\nAlpha:")
    print(alpha)

    #alpha -= torch.min(torch.min(alpha))
    #alpha *= 25500
    
    alpha_ = alpha.to('cpu')
    cam = alpha_.numpy()
    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam)
    cam = resize(cam, (224,224))
    cam = (cam*-1.0) + 1.0
    cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET))
    '''

